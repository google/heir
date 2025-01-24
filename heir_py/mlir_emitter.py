"""Emitter from numba IR (SSA) to MLIR."""

import operator
import textwrap
from dataclasses import dataclass
from collections import deque

from numba.core import ir


class HeaderInfo:
  """
  Header info provides info from a loop's header block

  Attributes:
    phi_var: An ir.Var representing the loop iterator
    body_id: The block id of the loop body
    next_id: The block id of the block after the loop
  """

  def __init__(self, header_block):
    body = header_block.body
    assert len(body) == 5
    self.phi_var = body[3].target
    self.body_id = body[4].truebr
    self.next_id = body[4].falsebr


class RangeArgs:
  """
  Range args provides a ranges start, stop, step args.

  Attributes:
    start: An ir.Var or int providing the lower bound
    stop: An ir.Var or int providing the upper bound
    step: An ir.Var or int providing the step
  """

  def __init__(self, range_call):
    args = range_call.value.args
    self.start = 0
    self.step = 1
    match len(args):
      case 1:
        self.stop = args
      case 2:
        self.start, self.stop = args
      case 3:
        self.start, self.stop, self.step = args


@dataclass
class Loop:
  """
  A dataclass containing Loop info.

  Attributes:
    header_id: The block id of the loop's header
    header: A HeaderInfo object with the iterator and loop block ids
    rage: A RangeArg object with the loop's range info
    inits: A list of ir.Vars that initialize any loop-carried vars
  """

  header_id: int
  header: HeaderInfo
  range: RangeArgs
  inits: list[ir.Var]


def build_loop_from_call(index, body, blocks):
  # Build a loop from a range call starting at index
  header_id = body[index + 3].target
  header = HeaderInfo(blocks[header_id])
  range_args = RangeArgs(body[index])

  # Loop body must start with assigning the local iter var
  loop_body = blocks[header.body_id].body
  assert loop_body[0].value == header.phi_var

  inits = []
  for instr in loop_body[1:]:
    if type(instr) == ir.Assign and not instr.target.is_temp:
      inits.append(instr.target)
  if len(inits) > 1:
    raise NotImplementedError("Multiple iter_args not supported")

  return Loop(
      header_id,
      header,
      range_args,
      inits,
  )


def is_start_of_loop(index, body, ssa_ir):
  instr = body[index]
  # True if instr is a range call that begins a loop
  match instr:
    case ir.Assign(value=ir.Expr(op="call")):
      func = instr.value.func
    case _:
      return False

  assert func.name in ssa_ir._definitions
  func_def_list = ssa_ir._definitions[func.name]
  assert len(func_def_list) == 1
  func_def = func_def_list[0]

  match func_def:
    case ir.Global(name="range"):
      # The next instrs should be getiter, iternext, and jump
      next_instrs = map(type, body[index + 1 : index + 4])
      if list(next_instrs) != [ir.Assign, ir.Assign, ir.Jump]:
        return False
    case _:
      return False

  return True


class TextualMlirEmitter:

  def __init__(self, ssa_ir):
    self.ssa_ir = ssa_ir
    self.temp_var_id = 0
    self.numba_names_to_ssa_var_names = {}
    self.globals_map = {}
    self.loops = {}

  def emit(self):
    func_name = self.ssa_ir.func_id.func_name
    # probably should use unique name...
    # func_name = ssa_ir.func_id.unique_name

    # TODO(#1162): use inferred or explicit types for args
    args_str = ", ".join([f"%{name}: i64" for name in self.ssa_ir.arg_names])

    # TODO(#1162): get inferred or explicit return types
    return_types_str = "i64"

    body = self.emit_blocks()

    mlir_func = f"""func.func @{func_name}({args_str}) -> ({return_types_str}) {{
{textwrap.indent(body, '  ')}
}}
"""
    return mlir_func

  def emit_blocks(self):
    blocks = self.ssa_ir.blocks

    # collect loops and block header needs
    block_ids_to_omit_header = set()
    for block_id, block in blocks.items():
      for i in range(len(block.body)):
        # Detect a range call
        instr = block.body[i]
        if is_start_of_loop(i, block.body, self.ssa_ir):
          loop = build_loop_from_call(i, block.body, blocks)
          self.loops[instr.target] = loop
          block_ids_to_omit_header.add(loop.header.next_id)

    sorted_blocks = sorted(blocks.items())
    # first block doesn't require a block header
    block_ids_to_omit_header.add(sorted_blocks[0][0])
    blocks_to_print = deque(sorted_blocks)

    # print blocks
    str_blocks = []
    while blocks_to_print:
      block_id, block = blocks_to_print.popleft()
      if block_id in block_ids_to_omit_header:
        block_header = ""
      else:
        block_header = f"^bb{block_id}:\n"
      str_blocks.append(
          block_header
          + textwrap.indent(self.emit_block(block, blocks_to_print), "  ")
      )

    return "\n".join(str_blocks)

  def emit_block(self, block, blocks_to_print):
    instructions = []
    for i in range(len(block.body)):
      instr = block.body[i]
      if type(instr) == ir.Assign and instr.target in self.loops:
        # We hit a range call for a loop
        result = self.emit_loop(instr.target, blocks_to_print)
        instructions.append(result)
        # Exit instructions, should be the end of the block
        break
      else:
        result = self.emit_instruction(instr)
      if result:
        instructions.append(result)
    return "\n".join(instructions)

  def emit_instruction(self, instr):
    match instr:
      case ir.Assign():
        return self.emit_assign(instr)
      case ir.Branch():
        return self.emit_branch(instr)
      case ir.Return():
        return self.emit_return(instr)
      case ir.Jump():
        # TODO ignore
        assert instr.is_terminator
        return
    raise NotImplementedError("Unsupported instruction: " + str(instr))

  def get_or_create_name(self, var):
    name = var.name
    if name in self.numba_names_to_ssa_var_names:
      ssa_id = self.numba_names_to_ssa_var_names[name]
    else:
      ssa_id = self.temp_var_id
      self.numba_names_to_ssa_var_names[name] = ssa_id
      self.temp_var_id += 1
    return f"%{ssa_id}"

  def forward_to_new_id(self, from_var):
    # Creates a new SSA var name and forwards the old var.
    # Useful to create new intermediate vars or temp vars.
    to_name = self.temp_var_id
    self.temp_var_id += 1
    self.numba_names_to_ssa_var_names[from_var.name] = to_name
    return f"%{to_name}"

  def get_name(self, var):
    assert var.name in self.numba_names_to_ssa_var_names
    return self.get_or_create_name(var)

  def forward_name(self, from_var, to_var):
    to_name = self.numba_names_to_ssa_var_names[to_var.name]
    self.numba_names_to_ssa_var_names[from_var.name] = to_name

  def emit_assign(self, assign):
    match assign.value:
      case ir.Arg():
        if assign.target.name != assign.value.name:
          raise ValueError(
              "MLIR has no vanilla assignment op? "
              "Do I need to keep a mapping from func arg names to SSA names?"
          )
        self.numba_names_to_ssa_var_names[assign.target.name] = (
            assign.value.name
        )
        return ""
      case ir.Expr(op="binop"):
        name = self.get_or_create_name(assign.target)
        emitted_expr = self.emit_binop(assign.value)
        # TODO(#1162): replace i64 with inferred type
        return f"{name} = {emitted_expr} : i64"
      case ir.Expr(op="call"):
        func = assign.value.func
        # if assert fails, variable was undefined
        assert func.name in self.globals_map
        if self.globals_map[func.name] == "bool":
          # nothing to do, forward the name to the arg of bool()
          self.forward_name(from_var=assign.target, to_var=assign.value.args[0])
          return ""
        else:
          raise NotImplementedError("Unknown global " + func.name)
      case ir.Expr(op="cast"):
        # not sure what to do here. maybe will be needed for type conversions
        # when interfacing with C
        self.forward_name(from_var=assign.target, to_var=assign.value.value)
        return ""
      case ir.Const():
        name = self.get_or_create_name(assign.target)
        # TODO(#1162): fix type (somehow the pretty printer on assign.value
        # knows it's an int???)
        return f"{name} = arith.constant {assign.value.value} : i64"
      case ir.Global():
        self.globals_map[assign.target.name] = assign.value.name
        return ""
      case ir.Var():
        self.forward_name(from_var=assign.target, to_var=assign.value)
        return ""
    raise NotImplementedError()

  def emit_expr(self, expr):
    if expr.op == "binop":
      return self.emit_binop(expr)
    elif expr.op == "inplace_binop":
      raise NotImplementedError()
    elif expr.op == "unary":
      raise NotImplementedError()
    elif expr.op == "pair_first":
      raise NotImplementedError()
    elif expr.op == "pair_second":
      raise NotImplementedError()
    elif expr.op in ("getiter", "iternext"):
      raise NotImplementedError()

    # these are all things numba has hooks for upstream, but we didn't implement
    # in the prototype

    elif expr.op == "exhaust_iter":
      raise NotImplementedError()
    elif expr.op == "getattr":
      raise NotImplementedError()
    elif expr.op == "static_getitem":
      raise NotImplementedError()
    elif expr.op == "typed_getitem":
      raise NotImplementedError()
    elif expr.op == "getitem":
      raise NotImplementedError()
    elif expr.op == "build_tuple":
      raise NotImplementedError()
    elif expr.op == "build_list":
      raise NotImplementedError()
    elif expr.op == "build_set":
      raise NotImplementedError()
    elif expr.op == "build_map":
      raise NotImplementedError()
    elif expr.op == "phi":
      raise ValueError("PHI not stripped")
    elif expr.op == "null":
      raise NotImplementedError()
    elif expr.op == "undef":
      raise NotImplementedError()

    raise NotImplementedError("Unsupported expr")

  def emit_binop(self, binop):
    lhs_ssa = self.get_name(binop.lhs)
    rhs_ssa = self.get_name(binop.rhs)

    match binop.fn:
      case operator.lt:
        return f"arith.cmpi slt, {lhs_ssa}, {rhs_ssa}"
      case operator.add:
        return f"arith.addi {lhs_ssa}, {rhs_ssa}"
      case operator.mul:
        return f"arith.muli {lhs_ssa}, {rhs_ssa}"
      case operator.sub:
        return f"arith.subi {lhs_ssa}, {rhs_ssa}"

    raise NotImplementedError("Unsupported binop: " + binop.fn.__name__)

  def emit_branch(self, branch):
    condvar = self.get_name(branch.cond)
    return f"cf.cond_br {condvar}, ^bb{branch.truebr}, ^bb{branch.falsebr}"

  def emit_var_or_int(self, var_or_int):
    if type(var_or_int) == ir.Var:
      # Create an index_cast operation
      var_name = self.get_name(var_or_int)
      new_name = self.forward_to_new_id(var_or_int)
      return (
          new_name,
          f"{new_name} = arith.index_cast {var_name} : i64 to index",
      )
    else:
      return var_or_int, ""

  def emit_loop_header(self, target, loop):
    # Returns the loop header str, e.g.
    # affine.for %i = 0 to 3 step 4 iter_args(%init = %var) {
    header = []
    resultvar = self.get_or_create_name(target)
    start, instr = self.emit_var_or_int(loop.range.start)
    if instr:
      header.append(instr)
    stop, instr = self.emit_var_or_int(loop.range.stop)
    if instr:
      header.append(instr)
    step, instr = self.emit_var_or_int(loop.range.step)
    if instr:
      header.append(instr)

    itvar = self.get_or_create_name(loop.header.phi_var)  # create var for i
    for_str = f"affine.for {itvar} = {start} to {stop}"
    if step != 1:
      for_str = f"{for_str} step {step}"

    if len(loop.inits) == 1:
      # Note: we must generalize to inits > 1
      init_val = self.get_name(loop.inits[0])
      # Within the loop, forward the name for the init val to a new temp var
      iter_arg = self.forward_to_new_id(loop.inits[0])
      for_str = (
          f"{resultvar} = {for_str} iter_args({iter_arg} = {init_val}) -> (i64)"
      )
    header.append(for_str + " {")
    return "\n".join(header)

  def emit_loop(self, target, blocks_to_print):
    loop = self.loops[target]
    header = self.emit_loop_header(target, loop)
    blocks_to_print.popleft()

    # TODO(#1412): support nested loops.
    body_id, loop_block = blocks_to_print.popleft()
    assert body_id == loop.header.body_id
    for instr in loop_block.body:
      if type(instr) == ir.Assign and instr.target in self.loops:
        raise NotImplementedError("Nested loops are not supported")

    body_str = self.emit_block(loop_block, blocks_to_print)
    if len(loop.inits) == 1:
      # Yield the iter arg
      yield_var = self.get_name(loop.inits[0])
      yield_str = f"affine.yield {yield_var} : i64"
      body_str += "\n" + yield_str

    # After we emit the body, we need to replace all uses of the iterarg after
    # the block with the result of the for loop.
    if loop.inits:
      self.forward_name(loop.inits[0], target)

    result = "\n".join([header, textwrap.indent(body_str, "  "), "}"])
    return result

  def emit_return(self, ret):
    var = self.get_name(ret.value)
    # TODO(#1162): replace i64 with inferred or explicit return type
    return f"func.return {var} : i64"
