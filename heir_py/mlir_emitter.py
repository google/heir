"""Emitter from numba IR (SSA) to MLIR."""

import operator
import textwrap
from dataclasses import dataclass

from numba.core import ir


def get_constant(var, ssa_ir):
  # Get constant value defining this var, else raise error
  assert var.name in ssa_ir._definitions
  vardef = ssa_ir._definitions[var.name][0]
  if type(vardef) != ir.Const:
    raise ValueError("expected constant variable")
  return vardef.value


class HeaderInfo:

  def __init__(self, header_block):
    body = header_block.body
    assert len(body) == 5
    self.phi_var = body[3].target
    self.body_id = body[4].truebr
    self.next_id = body[4].falsebr


class RangeArgs:

  def __init__(self, range_call, ssa_ir):
    args = range_call.value.args
    self.stop = get_constant(args[0], ssa_ir)
    self.start = 0
    self.step = 1
    if len(args) > 1:
      self.stop = get_constant(args[1], ssa_ir)
    if len(args) > 2:
      self.step = get_constant(args[2], ssa_ir)


@dataclass
class Loop:
  header_id: int
  header: HeaderInfo
  range: RangeArgs
  inits: list[str]


def build_loop_from_call(index, body, blocks, ssa_ir):
  # Build a loop from a range call starting at index
  assert type(body[index + 1] == ir.Assign)
  assert type(body[index + 2] == ir.Assign)
  assert type(body[index + 3] == ir.Jump)

  header_id = body[index + 3].target
  header = HeaderInfo(blocks[header_id])
  range_args = RangeArgs(body[index], ssa_ir)

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


def is_range_call(instr, ssa_ir):
  # Returns true if the IR instruction is a call to a range
  if type(instr) != ir.Assign or type(instr.value) != ir.Expr:
    return False
  if instr.value.op != "call":
    return False
  func = instr.value.func

  assert func.name in ssa_ir._definitions
  func_def_list = ssa_ir._definitions[func.name]
  assert len(func_def_list) == 1
  func_def = func_def_list[0]

  if type(func_def) != ir.Global:
    return False
  return func_def.name == "range"


class TextualMlirEmitter:

  def __init__(self, ssa_ir):
    self.ssa_ir = ssa_ir
    self.temp_var_id = 0
    self.numba_names_to_ssa_var_names = {}
    self.globals_map = {}
    self.loops = {}
    self.printed_blocks = {}
    self.omit_block_header = {}

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
    self.omit_block_header[sorted(blocks.items())[0][0]] = True
    for block_id, block in sorted(blocks.items()):
      for i in range(len(block.body)):
        # Detect a range call
        instr = block.body[i]
        if is_range_call(instr, self.ssa_ir):
          loop = build_loop_from_call(i, block.body, blocks, self.ssa_ir)
          self.loops[instr.target] = loop
          self.omit_block_header[loop.header.next_id] = True

    # print blocks
    str_blocks = []
    for block_id, block in sorted(blocks.items()):
      if block_id in self.printed_blocks:
        continue
      if block_id in self.omit_block_header:
        block_header = ""
      else:
        block_header = f"^bb{block_id}:\n"
      str_blocks.append(
          block_header + textwrap.indent(self.emit_block(block), "  ")
      )
      self.printed_blocks[block_id] = True

    return "\n".join(str_blocks)

  def emit_block(self, block):
    instructions = []
    for i in range(len(block.body)):
      instr = block.body[i]
      if type(instr) == ir.Assign and instr.target in self.loops:
        # We hit a range call for a loop
        result = self.emit_loop(instr.target)
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
        # FIXME: keep track of loop var, and assign
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

  def emit_loop(self, target):
    # Note right now loops that use the %i value directly will need an index_cast to an i64 element.
    loop = self.loops[target]
    resultvar = self.get_or_create_name(target)
    itvar = self.get_or_create_name(loop.header.phi_var)  # create var for i
    for_str = (
        f"affine.for {itvar} = {loop.range.start} to {loop.range.stop} step"
        f" {loop.range.step}"
    )

    if len(loop.inits) == 1:
      # Note: we must generalize to inits > 1
      init_val = self.get_name(loop.inits[0])
      # Within the loop, forward the name for the init val to a new temp var
      iter_arg = "itarg"
      self.numba_names_to_ssa_var_names[loop.inits[0].name] = iter_arg
      for_str = (
          f"{resultvar} = {for_str} iter_args(%{iter_arg} = {init_val}) ->"
          " (i64)"
      )
    self.printed_blocks[loop.header_id] = True

    # TODO(#1412): support nested loops.
    loop_block = self.ssa_ir.blocks[loop.header.body_id]
    for instr in loop_block.body:
      if type(instr) == ir.Assign and instr.target in self.loops:
        raise NotImplementedError("Nested loops are not supported")

    body_str = self.emit_block(loop_block)
    if len(loop.inits) == 1:
      # Yield the iter arg
      yield_var = self.get_name(loop.inits[0])
      yield_str = f"affine.yield {yield_var} : i64"
      body_str += "\n" + yield_str

    # After we emit the body, we need to update the printed block map and replace all uses of the iterarg after the block with the result of the for loop.
    if loop.inits:
      self.forward_name(loop.inits[0], target)
    self.printed_blocks[loop.header.body_id] = True

    result = for_str + " {\n" + textwrap.indent(body_str, "  ") + "\n}"
    return result

  def emit_return(self, ret):
    var = self.get_name(ret.value)
    # TODO(#1162): replace i64 with inferred or explicit return type
    return f"func.return {var} : i64"
