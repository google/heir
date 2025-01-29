"""Emitter from numba IR (SSA) to MLIR."""

import operator
import textwrap

from numba.core import ir
from numba.core import types

def mlirType(numba_type):
  if isinstance(numba_type, types.Integer):
    #TODO (#1162): fix handling of signedness
    # Since `arith` only allows signless integers, we ignore signedness here.
    return "i" + str(numba_type.bitwidth)
  if isinstance(numba_type, types.Boolean):
    return "i1"
  if isinstance(numba_type, types.Float):
    return "f" + str(numba_type.bitwidth)
  if isinstance(numba_type, types.Complex):
    return "complex<" + str(numba_type.bitwidth) + ">"
  if isinstance(numba_type, types.Array):
    #TODO (#1162): implement support for statically sized tensors
    # this probably requires extending numba with a new type
    # See https://numba.readthedocs.io/en/stable/extending/index.html
    return "tensor<" + "?x" * numba_type.ndim + mlirType(numba_type.dtype) +  ">"
  raise NotImplementedError("Unsupported type: " + str(numba_type))

def mlirLoc(loc : ir.Loc):
  return f"loc(\"{loc.filename or '<unknown>'}\":{loc.line or 0}:{loc.col or 0})"

def arithSuffix(numba_type):
  if isinstance(numba_type, types.Integer):
    return "i"
  if isinstance(numba_type, types.Boolean):
    return "i"
  if isinstance(numba_type, types.Float):
    return "f"
  if isinstance(numba_type, types.Complex):
    raise NotImplementedError("Complex numbers not supported in `arith` dialect")
  if isinstance(numba_type, types.Array):
    return arithSuffix(numba_type.dtype)
  raise NotImplementedError("Unsupported type: " + str(numba_type))


class TextualMlirEmitter:
  def __init__(self, ssa_ir, secret_args, typemap, retty):
    self.ssa_ir = ssa_ir
    self.secret_args = secret_args
    self.typemap = typemap
    self.retty = retty,
    self.temp_var_id = 0
    self.numba_names_to_ssa_var_names = {}
    self.globals_map = {}

  def emit(self):
    func_name = self.ssa_ir.func_id.func_name
    secret_flag = " {secret.secret}"
    # probably should use unique name...
    # func_name = ssa_ir.func_id.unique_name
    args_str = ", ".join([f"%{name}: {mlirType(self.typemap.get(name))}{secret_flag if idx in self.secret_args else str()} {mlirLoc(self.ssa_ir.loc)}" for idx, name in enumerate(self.ssa_ir.arg_names)])

    # TODO(#1162): support multiple return values!
    if(len(self.retty) > 1):
      raise NotImplementedError("Multiple return values not supported")
    return_types_str = mlirType(self.retty[0])

    body = self.emit_body()

    mlir_func = f"""func.func @{func_name}({args_str}) -> ({return_types_str}) {{
{textwrap.indent(body, '  ')}
}} {mlirLoc(self.ssa_ir.loc)}
"""
    return mlir_func

  def emit_body(self):
    blocks = self.ssa_ir.blocks
    str_blocks = []
    first = True

    for block_id, block in sorted(blocks.items()):
      instructions = []
      for instr in block.body:
        result = self.emit_instruction(instr)
        if result:
          instructions.append(result)

      if first:
        first = False
        block_header = ""
      else:
        block_header = f"^bb{block_id}:\n"

      str_blocks.append(
          block_header + textwrap.indent("\n".join(instructions), "  ")
      )

    return "\n".join(str_blocks)

  def emit_instruction(self, instr):
    match instr:
      case ir.Assign():
        return self.emit_assign(instr)
      case ir.Branch():
        return self.emit_branch(instr)
      case ir.Return():
        return self.emit_return(instr)
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

  def get_next_name(self):
    ssa_id = self.temp_var_id
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
        return f"{name} = {emitted_expr} : {mlirType(self.typemap.get(assign.target.name))} {mlirLoc(assign.loc)}"
      case ir.Expr(op="call"):
        func = assign.value.func
        # if assert fails, variable was undefined
        assert func.name in self.globals_map
        if self.globals_map[func.name] == "bool":
          # nothing to do, forward the name to the arg of bool()
          self.forward_name(from_var=assign.target, to_var=assign.value.args[0])
          return ""
        if self.globals_map[func.name] == "matmul":
          # emit `linalg.matmul` operation.
          lhs = self.get_name(assign.value.args[0])
          lhs_ty = mlirType(self.typemap.get(assign.value.args[0].name))
          rhs = self.get_name(assign.value.args[1])
          rhs_ty = mlirType(self.typemap.get(assign.value.args[1].name))
          target_numba_type = self.typemap.get(assign.target.name)
          out_ty = mlirType(target_numba_type)
          name = self.get_or_create_name(assign.target)
          if isinstance(target_numba_type, types.Array):
            # We need to emit a tensor.empty() operation to create the output tensor
            # but before that, we need to emit tensor.dim to get the sizes for the empty tensor
            str = ""
            dims = []
            for i in range(target_numba_type.ndim):
              cst = self.get_next_name()
              str += f"{cst} = arith.constant {i} : index {mlirLoc(assign.loc)}\n"
              dim = self.get_next_name()
              dims.append(dim)
              str += f"{dim} = tensor.dim {lhs}, {cst} : {lhs_ty} {mlirLoc(assign.loc)}\n"

            empty = self.get_next_name()
            str += f"{empty} = tensor.empty({','.join(dims)}) : {out_ty} {mlirLoc(assign.loc)}\n"
            str += f"{name} = linalg.matmul ins({lhs}, {rhs} : {lhs_ty}, {rhs_ty}) outs({empty} : {out_ty}) -> {out_ty} {mlirLoc(assign.loc)}"
            return str
          else:
            #TODO (#1162): implement support for statically sized tensors
            # this probably requires extending numba with a new type
            # See https://numba.readthedocs.io/en/stable/extending/index.html
            raise NotImplementedError(f"Unsupported target type {target_numba_type} for {assign.target.name}.")
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
        return f"{name} = arith.constant {assign.value.value} : i64 {mlirLoc(assign.loc)}"
      case ir.Global():
        self.globals_map[assign.target.name] = assign.value.name
        return ""
    raise NotImplementedError(f"Unsupported IR Element: {assign}")

  def emit_expr(self, expr):
    if expr.op == "binop":
      return self.emit_binop(expr)
    elif expr.op == "inplace_binop":
      raise NotImplementedError()
    elif expr.op == "unary":
      raise NotImplementedError()

    # these are all things numba has hooks for upstream, but we didn't implement
    # in the prototype

    elif expr.op == "pair_first":
      raise NotImplementedError()
    elif expr.op == "pair_second":
      raise NotImplementedError()
    elif expr.op in ("getiter", "iternext"):
      raise NotImplementedError()
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
    # This should be the same, otherwise MLIR will complain
    suffix = arithSuffix(self.typemap.get(str(binop.lhs)))

    match binop.fn:
      case operator.lt:
        return f"arith.cmp{suffix} slt, {lhs_ssa}, {rhs_ssa}"
      case operator.add:
        return f"arith.add{suffix} {lhs_ssa}, {rhs_ssa}"
      case operator.mul:
        return f"arith.mul{suffix} {lhs_ssa}, {rhs_ssa}"
      case operator.sub:
        return f"arith.sub{suffix} {lhs_ssa}, {rhs_ssa}"

    raise NotImplementedError("Unsupported binop: " + binop.fn.__name__)

  def emit_branch(self, branch):
    condvar = self.get_name(branch.cond)
    return f"cf.cond_br {condvar}, ^bb{branch.truebr}, ^bb{branch.falsebr}"

  def emit_return(self, ret):
    var = self.get_name(ret.value)
    return f"func.return {var} : {mlirType(self.typemap.get(str(ret.value)))} {mlirLoc(ret.loc)}"
