"""Emitter from numba IR (SSA) to MLIR."""

import operator
import textwrap

from numba.core import ir


class TextualMlirEmitter:
    def __init__(self, ssa_ir):
        self.ssa_ir = ssa_ir
        self.temp_var_id = 0
        self.numba_names_to_ssa_var_names = {}
        self.globals_map = {}

    def emit(self):
        func_name = self.ssa_ir.func_id.func_name
        # probably should use unique name...
        # func_name = ssa_ir.func_id.unique_name

        # TODO(#1162): use inferred or explicit types for args
        args_str = ", ".join([f"%{name}: i64" for name in self.ssa_ir.arg_names])

        # TODO(#1162): get inferred or explicit return types
        return_types_str = "i64"

        body = self.emit_body()

        mlir_func = f"""func.func @{func_name}({args_str}) -> ({return_types_str}) {{
{textwrap.indent(body, '  ')}
}}
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
                self.numba_names_to_ssa_var_names[
                    assign.target.name
                ] = assign.value.name
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
                    self.forward_name(
                        from_var=assign.target, to_var=assign.value.args[0]
                    )
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
                # TODO(#1162): fix type (somehow the pretty printer on assign.value knows it's
                # an int???)
                return f"{name} = arith.constant {assign.value.value} : i64"
            case ir.Global():
                self.globals_map[assign.target.name] = assign.value.name
                return ""
        raise NotImplementedError()

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

    def emit_return(self, ret):
        var = self.get_name(ret.value)
        # TODO(#1162): replace i64 with inferred or explicit return type
        return f"func.return {var} : i64"
