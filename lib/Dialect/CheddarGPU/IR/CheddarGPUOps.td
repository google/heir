#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_

include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.td"
include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class CheddarGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<CheddarGPU_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

// === Memory and Data Movement Operations ===

def CheddarGPU_CopyOp : CheddarGPU_Op<"copy"> {
  let summary = "Pointwise copy of a polynomial buffer or slice";
  let description = [{
    Copies elements from `in` to `out`.
    Supports optional `in_limb_range` and `out_limb_range` [start, count].
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "source buffer", [MemRead]>:$in,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out) `,` type($in)";
}

// === Binary and Unary Element-Wise Operations ===

def CheddarGPU_AddOp : CheddarGPU_Op<"add", [AllTypesMatch<["out", "lhs", "rhs"]>]> {
  let summary = "Pointwise modular addition of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `kernel::Sum` and `ElementWiseHandler::Add` in `ElementWise.cu`.
    Computes `out = lhs + rhs mod q` pointwise for each limb.

    Form contract:
    - Modular addition is valid in both `coeff` and `eval` domain.
    - Both operands must have matching form (`lhs_form == rhs_form == out_form`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "left-hand side operand", [MemRead]>:$lhs,
    Arg<AnyMemRef, "right-hand side operand", [MemRead]>:$rhs,
    CheddarGPU_FormAttr:$lhs_form,
    CheddarGPU_FormAttr:$rhs_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $lhs `,` $rhs prop-dict attr-dict `:` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$lhs, "Value":$rhs,
                   "FormAttr":$lhs_form, "FormAttr":$rhs_form, "FormAttr":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>,
    OpBuilder<(ins "Value":$out, "Value":$lhs, "Value":$rhs,
                   "Form":$lhs_form, "Form":$rhs_form, "Form":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>
  ];
}

def CheddarGPU_SubOp : CheddarGPU_Op<"sub", [AllTypesMatch<["out", "lhs", "rhs"]>]> {
  let summary = "Pointwise modular subtraction of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `kernel::Sub` and `ElementWiseHandler::Sub` in `ElementWise.cu`.
    Computes `out = lhs - rhs mod q` pointwise for each limb.

    Form contract:
    - Modular subtraction is valid in both `coeff` and `eval` domain.
    - Both operands must have matching form (`lhs_form == rhs_form == out_form`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "left-hand side operand", [MemRead]>:$lhs,
    Arg<AnyMemRef, "right-hand side operand", [MemRead]>:$rhs,
    CheddarGPU_FormAttr:$lhs_form,
    CheddarGPU_FormAttr:$rhs_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $lhs `,` $rhs prop-dict attr-dict `:` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$lhs, "Value":$rhs,
                   "FormAttr":$lhs_form, "FormAttr":$rhs_form, "FormAttr":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>,
    OpBuilder<(ins "Value":$out, "Value":$lhs, "Value":$rhs,
                   "Form":$lhs_form, "Form":$rhs_form, "Form":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>
  ];
}

def CheddarGPU_NegOp : CheddarGPU_Op<"neg", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Pointwise modular negation of a polynomial buffer";
  let description = [{
    Mirrors Cheddar's `kernel::Neg` and `ElementWiseHandler::Neg` in `ElementWise.cu`.
    Computes `out = -in mod q` pointwise for each limb.

    Form contract:
    - Valid in both `coeff` and `eval` domain.
    - Preserves representation form (`in_form == out_form`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$in,
                   "FormAttr":$in_form, "FormAttr":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$in_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, in, in_form, out_form,
            in_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>,
    OpBuilder<(ins "Value":$out, "Value":$in,
                   "Form":$in_form, "Form":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$in_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, in, in_form, out_form,
            in_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>
  ];
}

def CheddarGPU_Add2Op : CheddarGPU_Op<"add_2", [
    AllTypesMatch<["out_bx", "out_ax", "lhs_bx", "lhs_ax", "rhs_bx", "rhs_ax"]>]> {
  let summary = "Pointwise modular addition of two pairs of polynomial buffers";
  let description = [{
    Mirrors Cheddar's `Sum<Li2>` in `ElementWise.cu`.
    Computes `out_bx = lhs_bx + rhs_bx` and `out_ax = lhs_ax + rhs_ax` pointwise mod q.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "lhs bx buffer", [MemRead]>:$lhs_bx,
    Arg<AnyMemRef, "lhs ax buffer", [MemRead]>:$lhs_ax,
    Arg<AnyMemRef, "rhs bx buffer", [MemRead]>:$rhs_bx,
    Arg<AnyMemRef, "rhs ax buffer", [MemRead]>:$rhs_ax,
    CheddarGPU_FormAttr:$lhs_form,
    CheddarGPU_FormAttr:$rhs_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out_bx `,` $out_ax `,` $lhs_bx `,` $lhs_ax `,` $rhs_bx `,` $rhs_ax prop-dict attr-dict `:` type($out_bx)";

  let hasVerifier = 1;
}

def CheddarGPU_Sub2Op : CheddarGPU_Op<"sub_2", [
    AllTypesMatch<["out_bx", "out_ax", "lhs_bx", "lhs_ax", "rhs_bx", "rhs_ax"]>]> {
  let summary = "Pointwise modular subtraction of two pairs of polynomial buffers";
  let description = [{
    Mirrors Cheddar's `Sub<Li2>` in `ElementWise.cu`.
    Computes `out_bx = lhs_bx - rhs_bx` and `out_ax = lhs_ax - rhs_ax` pointwise mod q.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "lhs bx buffer", [MemRead]>:$lhs_bx,
    Arg<AnyMemRef, "lhs ax buffer", [MemRead]>:$lhs_ax,
    Arg<AnyMemRef, "rhs bx buffer", [MemRead]>:$rhs_bx,
    Arg<AnyMemRef, "rhs ax buffer", [MemRead]>:$rhs_ax,
    CheddarGPU_FormAttr:$lhs_form,
    CheddarGPU_FormAttr:$rhs_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out_bx `,` $out_ax `,` $lhs_bx `,` $lhs_ax `,` $rhs_bx `,` $rhs_ax prop-dict attr-dict `:` type($out_bx)";

  let hasVerifier = 1;
}

def CheddarGPU_Neg2Op : CheddarGPU_Op<"neg_2", [
    AllTypesMatch<["out_bx", "out_ax", "in_bx", "in_ax"]>]> {
  let summary = "Pointwise modular negation of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `Neg<Li2>` in `ElementWise.cu`.
    Computes `out_bx = -in_bx` and `out_ax = -in_ax` pointwise mod q.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "in bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "in ax buffer", [MemRead]>:$in_ax,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out_bx `,` $out_ax `,` $in_bx `,` $in_ax prop-dict attr-dict `:` type($out_bx)";

  let hasVerifier = 1;
}

def CheddarGPU_AddConstOp : CheddarGPU_Op<"add_const", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Pointwise addition of per-limb constants to a polynomial buffer";
  let description = [{
    Mirrors Cheddar's `kernel::AddConst` and `ElementWiseHandler::AddConst` in `ElementWise.cu`.
    Computes `out[l][i] = in[l][i] + cst[l] mod q_l`.

    Form contract:
    - Valid in both `coeff` and `eval` domain (`in_form == out_form`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<AnyMemRef, "per-limb constant buffer", [MemRead]>:$cst,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = "$out `,` $in `,` $cst prop-dict attr-dict `:` type($out) `,` type($cst)";
}

def CheddarGPU_SubConstOp : CheddarGPU_Op<"sub_const", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Pointwise subtraction of per-limb constants from a polynomial buffer";
  let description = [{
    Mirrors Cheddar's `kernel::SubConst` and `ElementWiseHandler::SubConst` in `ElementWise.cu`.
    Computes `out[l][i] = in[l][i] - cst[l] mod q_l`.

    Form contract:
    - Valid in both `coeff` and `eval` domain (`in_form == out_form`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<AnyMemRef, "per-limb constant buffer", [MemRead]>:$cst,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = "$out `,` $in `,` $cst prop-dict attr-dict `:` type($out) `,` type($cst)";
}

def CheddarGPU_SubOppositeConstOp : CheddarGPU_Op<"sub_opposite_const", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Pointwise subtraction of polynomial from per-limb constants";
  let description = [{
    Mirrors Cheddar's `kernel::SubOppositeConst` and `ElementWiseHandler::SubOppositeConst` in `ElementWise.cu`.
    Computes `out[l][i] = cst[l] - in[l][i] mod q_l`.

    Form contract:
    - Valid in both `coeff` and `eval` domain (`in_form == out_form`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<AnyMemRef, "per-limb constant buffer", [MemRead]>:$cst,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = "$out `,` $in `,` $cst prop-dict attr-dict `:` type($out) `,` type($cst)";
}

def CheddarGPU_MulOp : CheddarGPU_Op<"mul", [AllTypesMatch<["out", "lhs", "rhs"]>]> {
  let summary = "Pointwise Montgomery modular multiplication";
  let description = [{
    Mirrors Cheddar's `kernel::Mult` and `ElementWiseHandler::Mult` in `ElementWise.cu`.
    Computes `out = MultMontgomery(lhs, rhs, q, q_inv)`.

    Form contract:
    - Polynomial multiplication in evaluation domain is pointwise Montgomery multiplication:
      `(a * R) * (b * R) * R^-1 = (a * b) * R mod q`.
    - Operands MUST be in `eval` form (`lhs_form == rhs_form == out_form == eval`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "left-hand side operand", [MemRead]>:$lhs,
    Arg<AnyMemRef, "right-hand side operand", [MemRead]>:$rhs,
    CheddarGPU_FormAttr:$lhs_form,
    CheddarGPU_FormAttr:$rhs_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $lhs `,` $rhs prop-dict attr-dict `:` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$lhs, "Value":$rhs,
                   "FormAttr":$lhs_form, "FormAttr":$rhs_form, "FormAttr":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>,
    OpBuilder<(ins "Value":$out, "Value":$lhs, "Value":$rhs,
                   "Form":$lhs_form, "Form":$rhs_form, "Form":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/IntegerAttr{});
    }]>
  ];
}

def CheddarGPU_MulConstOp : CheddarGPU_Op<"mul_const", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Pointwise Montgomery multiplication by per-limb constants";
  let description = [{
    Mirrors Cheddar's `ElementWiseHandler::MultConst` in `ElementWise.cu`.
    Computes `out[l][i] = MultMontgomery(in[l][i], cst[l], q_l, q_inv_l)`.

    Form contract:
    - Montgomery multiplication requires `eval` form (`in_form == out_form == eval`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<AnyMemRef, "per-limb constant buffer", [MemRead]>:$cst,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $in `,` $cst prop-dict attr-dict `:` type($out) `,` type($cst)";

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$in, "Value":$cst,
                   "FormAttr":$in_form, "FormAttr":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$in_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range,
                   CArg<"IntegerAttr", "{}">:$prime_offset), [{
      build($_builder, $_state, out, in, cst, in_form, out_form,
            in_limb_range, out_limb_range, prime_offset, /*num_aux=*/IntegerAttr{});
    }]>,
    OpBuilder<(ins "Value":$out, "Value":$in, "Value":$cst,
                   "Form":$in_form, "Form":$out_form,
                   CArg<"DenseI64ArrayAttr", "{}">:$in_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range,
                   CArg<"IntegerAttr", "{}">:$prime_offset), [{
      build($_builder, $_state, out, in, cst, in_form, out_form,
            in_limb_range, out_limb_range, prime_offset, /*num_aux=*/IntegerAttr{});
    }]>
  ];
}

// === Accumulation Operations ===

def CheddarGPU_CPAccumOp : CheddarGPU_Op<"cp_accum", [AttrSizedOperandSegments]> {
  let summary = "Accumulate Montgomery products: dst = sum_k (common_srcs[k] * ct_srcs[k])";
  let description = [{
    Mirrors Cheddar's `kernel::CPAccum` and `ElementWiseHandler::CPAccumWorker<false/true>`
    in `ElementWise.cu`.
    Computes `out = sum_k MultMontgomery(ct_srcs[k], common_srcs[k])`.
    `common_srcs` can be polynomial buffers (`memref<LxNxiW>`) or per-limb constants (`memref<LxiW>`).

    Form contract:
    - Montgomery pointwise multiplication requires `eval` form for all operands.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Variadic<AnyMemRef>:$ct_srcs,
    Variadic<AnyMemRef>:$common_srcs,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<ArrayAttr>:$ct_forms,
    OptionalAttr<ArrayAttr>:$common_forms,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = [{
    $out `,` `ct_srcs` `(` $ct_srcs `)` `,` `common_srcs` `(` $common_srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

def CheddarGPU_CPAccum2Op : CheddarGPU_Op<"cp_accum_2", [AttrSizedOperandSegments]> {
  let summary = "Accumulate Montgomery products for two polynomials: {out_bx, out_ax} = sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])";
  let description = [{
    Mirrors Cheddar's `CPAccum<word,2,...>` / `PAccum` in `Context.cpp` / `ElementWise.cu`.
    Computes `{out_bx, out_ax} = sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])` over `k < K` (K <= 4),
    reading each `common[k]` (mod-up digit) once for both outputs.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Variadic<AnyMemRef>:$ct_bx_srcs,
    Variadic<AnyMemRef>:$ct_ax_srcs,
    Variadic<AnyMemRef>:$common_srcs,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<ArrayAttr>:$ct_forms,
    OptionalAttr<ArrayAttr>:$common_forms,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` `ct_bx_srcs` `(` $ct_bx_srcs `)` `,` `ct_ax_srcs` `(` $ct_ax_srcs `)` `,` `common_srcs` `(` $common_srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_CPAccumAddOp : CheddarGPU_Op<"cp_accum_add", [AttrSizedOperandSegments]> {
  let summary = "Accumulate Montgomery products with base: dst = acc + sum_k (common_srcs[k] * ct_srcs[k])";
  let description = [{
    Mirrors Cheddar's `kernel::CPAccumAdd` in `ElementWise.cu`.
    Computes `out = acc + sum_k MultMontgomery(ct_srcs[k], common_srcs[k])`.

    Form contract:
    - Montgomery pointwise multiplication requires `eval` form for all operands.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "initial accumulator buffer", [MemRead]>:$acc,
    Variadic<AnyMemRef>:$ct_srcs,
    Variadic<AnyMemRef>:$common_srcs,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<CheddarGPU_FormAttr>:$acc_form,
    OptionalAttr<ArrayAttr>:$ct_forms,
    OptionalAttr<ArrayAttr>:$common_forms,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = [{
    $out `,` `acc` `(` $acc `)` `,` `ct_srcs` `(` $ct_srcs `)` `,` `common_srcs` `(` $common_srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

def CheddarGPU_CPAccumAdd2Op : CheddarGPU_Op<"cp_accum_add_2", [AttrSizedOperandSegments]> {
  let summary = "Accumulate Montgomery products with base for two polynomials: {out_bx, out_ax} = {acc_bx, acc_ax} + sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])";
  let description = [{
    Mirrors Cheddar's `CPAccumAdd<L,2>` / `CAccum` in `Context.cpp` / `ElementWise.cu`.
    Computes `{out_bx, out_ax} = {acc_bx, acc_ax} + sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "initial accumulator bx buffer", [MemRead]>:$acc_bx,
    Arg<AnyMemRef, "initial accumulator ax buffer", [MemRead]>:$acc_ax,
    Variadic<AnyMemRef>:$ct_bx_srcs,
    Variadic<AnyMemRef>:$ct_ax_srcs,
    Variadic<AnyMemRef>:$common_srcs,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<CheddarGPU_FormAttr>:$acc_form,
    OptionalAttr<ArrayAttr>:$ct_forms,
    OptionalAttr<ArrayAttr>:$common_forms,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` `acc` `(` $acc_bx `,` $acc_ax `)` `,` `ct_bx_srcs` `(` $ct_bx_srcs `)` `,` `ct_ax_srcs` `(` $ct_ax_srcs `)` `,` `common_srcs` `(` $common_srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// === Ciphertext Tensor Operations ===

def CheddarGPU_TensorOp : CheddarGPU_Op<"tensor", [
    AllTypesMatch<["out_bx", "out_ax", "out_rx", "src1_bx", "src1_ax", "src2_bx", "src2_ax"]>]> {
  let summary = "Ciphertext tensor product: (b1*b2, b1*a2 + a1*b2, a1*a2)";
  let description = [{
    Mirrors Cheddar's `kernel::Tensor` and `ElementWiseHandler::Tensor` in `ElementWise.cu`.
    Multiplies two degree-1 ciphertexts (b1, a1) and (b2, a2) into a degree-2 ciphertext
    (b1*b2, b1*a2 + a1*b2, a1*a2) using Karatsuba modular multiplication.

    Form contract:
    - Requires `eval` form for all operands.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "destination rx buffer", [MemWrite]>:$out_rx,
    Arg<AnyMemRef, "src1 bx buffer", [MemRead]>:$src1_bx,
    Arg<AnyMemRef, "src1 ax buffer", [MemRead]>:$src1_ax,
    Arg<AnyMemRef, "src2 bx buffer", [MemRead]>:$src2_bx,
    Arg<AnyMemRef, "src2 ax buffer", [MemRead]>:$src2_ax,
    CheddarGPU_FormAttr:$lhs_form,
    CheddarGPU_FormAttr:$rhs_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $out_rx `,` $src1_bx `,` $src1_ax `,` $src2_bx `,` $src2_ax prop-dict attr-dict
    `:` type($out_bx)
  }];
}

def CheddarGPU_TensorSquareOp : CheddarGPU_Op<"tensor_square", [
    AllTypesMatch<["out_bx", "out_ax", "out_rx", "src_bx", "src_ax"]>]> {
  let summary = "Ciphertext tensor squaring: (b1^2, 2*b1*a1, a1^2)";
  let description = [{
    Mirrors Cheddar's `kernel::TensorSquare` in `ElementWise.cu`.
    Squares degree-1 ciphertext (b1, a1) into degree-2 ciphertext (b1^2, 2*b1*a1, a1^2).

    Form contract:
    - Requires `eval` form for all operands.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "destination rx buffer", [MemWrite]>:$out_rx,
    Arg<AnyMemRef, "src bx buffer", [MemRead]>:$src_bx,
    Arg<AnyMemRef, "src ax buffer", [MemRead]>:$src_ax,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $out_rx `,` $src_bx `,` $src_ax prop-dict attr-dict
    `:` type($out_bx)
  }];
}

// === Imaginary Unit and Automorphism Operations ===

def CheddarGPU_MulImaginaryUnitOp : CheddarGPU_Op<"mul_imag_unit", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Pointwise multiplication by imaginary unit in evaluation domain";
  let description = [{
    Mirrors Cheddar's `kernel::MultImaginaryUnit` in `ElementWise.cu`.
    In evaluation domain, multiplying by the imaginary unit corresponds to multiplying
    first half by `psi^(N/2)` and second half by `-psi^(N/2) = q - psi^(N/2)`.

    Form contract:
    - Requires `eval` form (`in_form == out_form == eval`).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<AnyMemRef, "per-limb imaginary unit table (psi^(N/2))", [MemRead]>:$i_unit,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range
  );

  let assemblyFormat = "$out `,` $in `,` $i_unit prop-dict attr-dict `:` type($out) `,` type($i_unit)";
}

def CheddarGPU_PermuteOp : CheddarGPU_Op<"permute", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Galois automorphism permutation in bit-reversed evaluation domain";
  let description = [{
    Mirrors Cheddar's `kernel::Permute` and `ElementWiseHandler::Permute` in `ElementWise.cu`.
    Computes automorphism gather index:
      `e = brev_{logN+1}(x) + 1`
      `src = brev_{logN+1}(e * g - 1)`
    where `g` is either the Galois factor (`galois` attr, `5^k mod 2N`) or complex conjugation
    `2N - 1` (`conj` unit attr).

    Form contract:
    - Automorphism gather math requires bit-reversed evaluation order (`eval` form).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<I64Attr>:$galois,
    UnitAttr:$conj,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out)";
}

def CheddarGPU_Permute2Op : CheddarGPU_Op<"permute_2", [
    AllTypesMatch<["out_bx", "out_ax", "in_bx", "in_ax"]>]> {
  let summary = "Galois automorphism permutation for two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `Permute<Li2>` in `ElementWise.cu`.
    Computes bit-reversed Galois automorphism gather on both bx and ax simultaneously.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input ax buffer", [MemRead]>:$in_ax,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<I64Attr>:$galois,
    UnitAttr:$conj,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out_bx `,` $out_ax `,` $in_bx `,` $in_ax prop-dict attr-dict `:` type($out_bx)";

  let hasVerifier = 1;
}

def CheddarGPU_PermuteAccumOp : CheddarGPU_Op<"permute_accum", [AttrSizedOperandSegments]> {
  let summary = "Accumulate multiple Galois automorphism permutations";
  let description = [{
    Mirrors Cheddar's `kernel::PermuteAccum` and `kernel::PermuteAccumAdd` in `ElementWise.cu`.
    Computes `out = (acc +) sum_k permute(srcs[k], galois_factors[k])`.

    Form contract:
    - Requires `eval` form for all operands.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Optional<AnyMemRef>:$acc,
    Variadic<AnyMemRef>:$srcs,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<ArrayAttr>:$in_forms,
    DenseI64ArrayAttr:$galois_factors,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = [{
    $out `,` (`acc` `(` $acc^ `)` `,`)? `srcs` `(` $srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

def CheddarGPU_PermuteAccum2Op : CheddarGPU_Op<"permute_accum_2", [AttrSizedOperandSegments]> {
  let summary = "Accumulate multiple Galois automorphism permutations for two polynomials";
  let description = [{
    Mirrors Cheddar's `PermuteAccum` for 2 polynomials in `ElementWise.cu`.
    Computes `{out_bx, out_ax} = ({acc_bx, acc_ax} +) sum_k permute({srcs_bx[k], srcs_ax[k]}, galois_factors[k])`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination ax buffer", [MemWrite]>:$out_ax,
    Optional<AnyMemRef>:$acc_bx,
    Optional<AnyMemRef>:$acc_ax,
    Variadic<AnyMemRef>:$srcs_bx,
    Variadic<AnyMemRef>:$srcs_ax,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<ArrayAttr>:$in_forms,
    DenseI64ArrayAttr:$galois_factors,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` (`acc` `(` $acc_bx^ `,` $acc_ax `)` `,`)? `srcs_bx` `(` $srcs_bx `)` `,` `srcs_ax` `(` $srcs_ax `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// === Number Theoretic Transform Operations ===

def CheddarGPU_NTTOp : CheddarGPU_Op<"ntt", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Forward Number Theoretic Transform (NTT)";
  let description = [{
    Mirrors Cheddar's `kernel::NTT` and `NTTHandler::NTT` in `NTT.cu` (~440-690)
    and `NTTUtils.cuh` (19-35).

    Mathematical contract:
    - Algorithm: Radix-2 Cooley-Tukey Decimation-in-Time (CT-DIT) butterfly network.
    - Butterfly formula (`NTTUtils.cuh:19-35`):
        mult = MontReduceLazy(b * w, q, q_inv)   // in (-q, q)
        if (mult < 0) mult += q                  // in [0, q)
        if (a < 0) a += q                        // in [0, q)
        b_new = a - mult                         // in (-q, q)
        a_new = (a - q) + mult                   // in [-q, q)
    - Twiddles: uses bit-reversed Montgomery twiddle table `psi_rev_mont[m + i]`
      where `psi = g^((q-1)/2N)` with smallest primitive root generator `g`.
    - Input ordering and domain: standard natural order in coefficient domain (`in_form = coeff`).
    - Output ordering and domain: bit-reversed order in Montgomery evaluation domain (`out_form = eval`).
    - Montgomery conversion semantics (`NTT.cu:642-648`):
      - When `montgomery_conversion == true`: input elements are standard residues
        `x in [0, q)`. They are premultiplied by `R^2 mod q` via Montgomery reduction:
        `(x * R^2) * R^-1 = x * R mod q`, converting them into Montgomery form before butterflies.
      - When `montgomery_conversion == false`: input elements are assumed to already be
        in Montgomery form (`x * R mod q`).
    - Final canonicalization: lazy internal range `[-q, q)` is canonicalized via
      `if (x < 0) x += q` to `[0, q)` at the end of the transform.
    - Optional `skip_range`: `[skip_start, skip_end]` for `NTTForModUp` semantics (`NTT.cu:815-860`),
      skipping transformation for limbs in `[skip_start, skip_end)`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<Optional<AnyMemRef>, "verbatim copy source buffer for skipped limbs", [MemRead]>:$skip_copy_src,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    BoolAttr:$montgomery_conversion,
    OptionalAttr<DenseI64ArrayAttr>:$skip_range,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$prime_range,
    OptionalAttr<DenseI64ArrayAttr>:$prime_indices,
    DefaultValuedAttr<I64Attr, "0">:$skip_copy_in_offset
  );

  let assemblyFormat = [{
    $out `,` $in ( `,` `skip_copy_src` `(` $skip_copy_src^ `)` )? prop-dict attr-dict `:` type($out) ( `,` type($skip_copy_src)^ )?
  }];

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$in,
                   "FormAttr":$in_form, "FormAttr":$out_form,
                   "BoolAttr":$montgomery_conversion,
                   CArg<"DenseI64ArrayAttr", "{}">:$skip_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$in_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$prime_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$prime_indices), [{
      build($_builder, $_state, out, in, /*skip_copy_src=*/Value(),
            in_form, out_form, montgomery_conversion, skip_range,
            in_limb_range, out_limb_range, prime_range, prime_indices,
            /*skip_copy_in_offset=*/0);
    }]>
  ];
}

def CheddarGPU_INTTOp : CheddarGPU_Op<"intt", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Inverse Number Theoretic Transform (INTT) with optional epilogue";
  let description = [{
    Mirrors Cheddar's `kernel::INTT`, `NTTHandler::INTT`, and `NTTHandler::INTTAndMultConst`
    in `NTT.cu` (~695-812) and `NTTUtils.cuh` (46-60, 363-391).

    Mathematical contract:
    - Algorithm: Radix-2 Gentleman-Sande Decimation-in-Frequency (GS-DIF) butterfly network.
    - Butterfly formula (`NTTUtils.cuh:46-60`):
        if (a < 0) a += q                        // in [0, q)
        if (b < 0) b += q                        // in [0, q)
        diff = a - b                             // in (-q, q)
        a_new = (a - q) + b                      // in [-q, q)
        b_new = MontReduceLazy(diff * w, q, q_inv) // in (-q, q)
    - Twiddles: uses bit-reversed inverse Montgomery twiddle table `psi_inv_rev_mont[m + i]`
      where `psi_inv = psi^-1 mod q`.
    - Input ordering and domain: bit-reversed order in evaluation domain (`in_form = eval`).
    - Output ordering and domain: standard natural order in coefficient domain.
    - Butterfly result accumulates scaling factor: `(x * R * N) mod q`.
    - Montgomery conversion and normalization (`NTT.cu:720-724`, `NTTUtils.cuh:363-391`):
      - When `epilogue == none`:
        - `montgomery_conversion == true` (Cheddar default): multiplies by plain `N^-1 mod q`
          via Montgomery reduction `MultConst` with twiddle `N^-1`:
          `(x * R * N) * N^-1 * R^-1 = x mod q`, leaving the result in standard
          (non-Montgomery) canonical representation in `[0, q)` (`out_form = coeff`).
        - `montgomery_conversion == false`: multiplies by `(N^-1 * R) mod q` via
          `MultConst`: `(x * R * N) * (N^-1 * R) * R^-1 = x * R mod q`, leaving the result
          in Montgomery form in `[0, q)` (`out_form = coeff`).
      - When `epilogue == mult_const` (`NTTHandler::INTTAndMultConst`, `NTT.cu:788-812`):
        - Multiplies the normalized INTT output by per-limb constant `c` via Montgomery reduction:
          `(x * c) * R^-1 mod q`, producing canonical residue in `[0, q)` (`out_form = coeff`).
      - When `epilogue == mult_const_normalize`:
        - Multiplies by per-limb constant `c` via Montgomery reduction, then centers the residue:
          `if (temp > (q >> 1)) temp -= q`, producing signed centered coefficients in
          `[-(q-1)/2, (q-1)/2]` stored as two's complement signless integers (`out_form = coeff_centered`).

    Note on ModDown variants:
    - Fused `NTTForModDown` and `INTTForModDown` involve cross-basis operations with dual
      modulus chains and are scheduled for Task T4.2.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    Arg<Optional<AnyMemRef>, "per-limb constant buffer", [MemRead]>:$cst,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    BoolAttr:$montgomery_conversion,
    CheddarGPU_INTTEpilogueAttr:$epilogue,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$prime_range,
    OptionalAttr<DenseI64ArrayAttr>:$prime_indices,
    OptionalAttr<DenseI64ArrayAttr>:$in_split
  );

  let assemblyFormat = [{
    $out `,` $in ( `,` $cst^ )? prop-dict attr-dict `:` type($out) ( `,` type($cst)^ )?
  }];

  let builders = [
    OpBuilder<(ins "Value":$out, "Value":$in, "Value":$cst,
                   "FormAttr":$in_form, "FormAttr":$out_form,
                   "BoolAttr":$montgomery_conversion,
                   "INTTEpilogueAttr":$epilogue,
                   CArg<"DenseI64ArrayAttr", "{}">:$in_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$out_limb_range,
                   CArg<"DenseI64ArrayAttr", "{}">:$prime_range), [{
      build($_builder, $_state, out, in, cst,
            in_form, out_form, montgomery_conversion, epilogue,
            in_limb_range, out_limb_range, prime_range,
            /*prime_indices=*/nullptr, /*in_split=*/nullptr);
    }]>
  ];
}

def CheddarGPU_BConvOp : CheddarGPU_Op<"bconv"> {
  let summary = "Approximate basis conversion (BConv) of polynomial buffers";
  let description = [{
    Mirrors Cheddar's `kernel::ModSwitchMatrixMult` in `ModSwitch.cu:24-170`.
    Converts polynomial coefficients from a source modulus chain (`in`, with `src_len` limbs)
    to a destination modulus chain (`out`, with `dst_len` computed limbs) via the approximate HPS
    (Halevi-Polyakov-Smart) basis conversion formula:
        out[m, x] = MontReduce(sum_{s=0}^{src_len-1} in[s, x] * table[m, s])
    where `table[m, s] = Normalize((q̂_s * R^2) mod p_m, p_m)` is the precomputed double-Montgomery
    projection table.

    Contract & Form Invariants:
    - Input form MUST be `coeff_centered`: input coefficients are signed residues in
      `[-(q_s-1)/2, (q_s-1)/2]` produced by an INTT with `mult_const_normalize` epilogue.
    - Output form MUST be `coeff`: output coefficients are standard canonical residues in
      `[0, p_m)`.
    - `table` is an explicit operand memref of shape `<DxSxiW>` where `D` is the number of
      computed destination limbs (`dst_len`) and `S` is the number of active source limbs (`src_len`).
    - `skip_range = [skip_start, skip_end]` is optional (used in ModUp where source limbs that are
      copied verbatim from source to destination are skipped by BConv). When specified, destination
      limbs `m` in `[0, dst_len)` map to limb index `(m < skip_start) ? m : m + (skip_end - skip_start)`
      in the destination buffer.
    - `in_limb_range` and `out_limb_range` are optional `[offset, count]` ranges for slicing larger buffers.

    Proof of Value-Preserving Periodic Re-centering:
    Cheddar accumulates products `in[s, x] * table[m, s]` in a 64-bit signed integer and periodically
    (every 4 terms) re-centers by adding/subtracting `prime_th = p_m * 2^W = p_m * R` when
    `accum >= prime_th / 2` or `accum < 0`.
    Because the final operation is Montgomery reduction:
        MontReduce(accum) = (accum * R^-1) mod p_m
    any integer multiple of `prime_th` satisfies:
        (k * prime_th) * R^-1 = (k * p_m * R) * R^-1 = k * p_m ≡ 0 (mod p_m).
    Therefore, periodic re-centering is value-preserving under Montgomery reduction: it yields the
    exact same canonical residue mod p_m as exact wide accumulation without re-centering, while
    bounding `accum` in `[-p_m * R / 2, p_m * R / 2)` to prevent 64-bit overflow and guarantee the
    range contract `(-p_m * R, p_m * R)` of lazy Montgomery reduction.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "source polynomial buffer", [MemRead]>:$in,
    Arg<AnyMemRef, "precomputed bconv matrix table memref<DxSxiW>", [MemRead]>:$table,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$skip_range,
    OptionalAttr<DenseI64ArrayAttr>:$in_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<DenseI64ArrayAttr>:$dst_primes
  );

  let assemblyFormat = [{
    $out `,` $in `,` $table prop-dict attr-dict `:` type($out) `,` type($in) `,` type($table)
  }];
}

// === ModSwitch Macro and Epilogue Operations ===

def CheddarGPU_PseudoModUpOp : CheddarGPU_Op<"pseudo_mod_up", [AllTypesMatch<["out", "in"]>]> {
  let summary = "Scales a polynomial buffer by p_prod = prod(P) mod q";
  let description = [{
    Mirrors Cheddar's `ModSwitchHandler::PseudoModUp` in `ModSwitch.cu:303-317`.
    Computes `out[l][i] = in[l][i] * p_prod[l] mod q_l` for each active limb.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out)";
}

def CheddarGPU_ModUpOp : CheddarGPU_Op<"mod_up"> {
  let summary = "ModUp basis extension: expands polynomial across beta destination buffers";
  let description = [{
    Mirrors Cheddar's `ModSwitchHandler::ModUp` in `ModSwitch.cu:318-398`.
    Expands input `in` (with `num_q` limbs) into `beta` destination buffers `outs`
    (each having `num_q + num_aux` limbs).
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<Variadic<AnyMemRef>, "destination buffers for each beta digit", [MemWrite]>:$outs,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    `(` $outs `)` `,` $in prop-dict attr-dict `:` `(` type($outs) `)` `,` type($in)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_ModDownOp : CheddarGPU_Op<"mod_down"> {
  let summary = "ModDown modulus switching: drops auxiliary primes P";
  let description = [{
    Mirrors Cheddar's `ModSwitchHandler::ModDown` in `ModSwitch.cu:400-416`.
    Shrinks input `in` (with `num_q + num_aux` limbs) down to `out` (with `num_q` limbs).
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out) `,` type($in)";
}

def CheddarGPU_RescaleOp : CheddarGPU_Op<"rescale"> {
  let summary = "Rescale modulus switching: drops terminal/main prime(s) from level to level - 1";
  let description = [{
    Mirrors Cheddar's `ModSwitchHandler::Rescale` in `ModSwitch.cu:418-435`.
    Shrinks input `in` (at `level`) down to `out` (at `level - 1`).
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out) `,` type($in)";
}

def CheddarGPU_ModDownRescaleOp : CheddarGPU_Op<"mod_down_rescale"> {
  let summary = "Combined ModDown and Rescale modulus switching";
  let description = [{
    Mirrors Cheddar's `ModSwitchHandler::ModDownAndRescale` in `ModSwitch.cu:437-453`.
    Shrinks input `in` (with `num_q(level) + num_aux` limbs) down to `out` (at `level - 1`).
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer", [MemRead]>:$in,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out) `,` type($in)";
}

def CheddarGPU_ModDownEpilogueOp : CheddarGPU_Op<"mod_down_epilogue", [AttrSizedOperandSegments]> {
  let summary = "Fused epilogue for ModDown, Rescale, and ModDownAndRescale";
  let description = [{
    Mirrors Cheddar's `NTTPhase2ForModDown` fused epilogue in `NTT.cu:457-601`.
    Computes:
      out[l, i] = ((src2_exists ? (src2[l - pad_start, i] * src2_padding[l - pad_start]) : 0)
                   - ntt_bconv[l, i]) * inv_prime_prod[l] mod q_l.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "ntt_bconv polynomial buffer", [MemRead]>:$ntt_bconv,
    Arg<AnyMemRef, "per-limb inv_prime_prod constants memref<LxiW>", [MemRead]>:$inv_prime_prod,
    Arg<Optional<AnyMemRef>, "optional src2 buffer for rescale/mod_down", [MemRead]>:$src2,
    Arg<Optional<AnyMemRef>, "optional src2_padding constants", [MemRead]>:$src2_padding,
    I64Attr:$pad_start,
    I64Attr:$pad_end,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out `,` $ntt_bconv `,` $inv_prime_prod ( `,` $src2^ ( `,` $src2_padding^ )? )? prop-dict attr-dict `:` type($out) `,` type($ntt_bconv) `,` type($inv_prime_prod) ( `,` type($src2)^ ( `,` type($src2_padding)^ )? )?
  }];
}

def CheddarGPU_NTTModDownOp : CheddarGPU_Op<"ntt_mod_down", [AttrSizedOperandSegments]> {
  let summary = "Fused Forward NTT and ModDown epilogue";
  let description = [{
    Mirrors Cheddar's `NTTPhase2ForModDown` in `NTT.cu:457-601`.
    Computes forward NTT on `ntt_in` followed by fused ModDown epilogue:
      out[l, x] = MontMul(ModSub(term1, NTT(ntt_in)[l, x]), inv_prime_prod[l]) mod q_l
    where
      term1 = 0                                                            (no src2)
            | src2[l + src2_offset, x]                                     (src2, l in [pad_start, pad_end))
            | MontMul(src2[l + src2_offset, x], src2_padding[l - pad_start]) (src2_pad, l in [pad_start, pad_end))
      and 0 outside [pad_start, pad_end).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "input polynomial buffer to transform (coeff form)", [MemRead]>:$ntt_in,
    Arg<AnyMemRef, "per-limb inv_prime_prod constants memref<LxiW>", [MemRead]>:$inv_prime_prod,
    Arg<Optional<AnyMemRef>, "optional src2 buffer", [MemRead]>:$src2,
    Arg<Optional<AnyMemRef>, "optional src2_padding constants", [MemRead]>:$src2_padding,
    I64Attr:$pad_start,
    I64Attr:$pad_end,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    DefaultValuedAttr<I64Attr, "0">:$src2_offset,
    DefaultValuedAttr<I64Attr, "0">:$prime_offset,
    OptionalAttr<DenseI64ArrayAttr>:$prime_indices
  );

  let assemblyFormat = [{
    $out `,` $ntt_in `,` $inv_prime_prod ( `,` $src2^ ( `,` $src2_padding^ )? )? prop-dict attr-dict `:` type($out) `,` type($ntt_in) `,` type($inv_prime_prod) ( `,` type($src2)^ ( `,` type($src2_padding)^ )? )?
  }];

  let hasVerifier = 1;
}

// === Key Switching Operations ===

def CheddarGPU_KeySwitchNoModDownOp : CheddarGPU_Op<"key_switch_no_mod_down", [AttrSizedOperandSegments]> {
  let summary = "Key switch without ModDown: ModUp + CPAccum + p_prod addition";
  let description = [{
    Computes destination-passing KeySwitch without modulus down:
      out_bx = sum_k (mod_up(in)_k * evk_bx_k) + (in_bx ? in_bx * p_prod : 0)
      out_ax = sum_k (mod_up(in)_k * evk_ax_k) + (in_ax ? in_ax * p_prod : 0)
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input polynomial buffer to key switch", [MemRead]>:$in,
    Arg<Optional<AnyMemRef>, "optional input in_bx to add", [MemRead]>:$in_bx,
    Arg<Optional<AnyMemRef>, "optional input in_ax to add", [MemRead]>:$in_ax,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in ( `in_bx` `(` $in_bx^ `)` )? ( `in_ax` `(` $in_ax^ `)` )? `,`
    `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)` prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_KeySwitchOp : CheddarGPU_Op<"key_switch", [AttrSizedOperandSegments]> {
  let summary = "KeySwitch with ModDown";
  let description = [{
    Performs key switching on input ciphertext `(in_bx, in_ax)` using evaluation key
    `(evk_bx, evk_ax)` and produces output ciphertext `(out_bx, out_ax)` at the same level.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)`
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_RelinearizeOp : CheddarGPU_Op<"relinearize", [AttrSizedOperandSegments]> {
  let summary = "Relinearize degree-2 ciphertext";
  let description = [{
    Relinearizes degree-2 ciphertext `(in_bx, in_ax, in_rx)` using relinearization key
    `(evk_bx, evk_ax)` producing degree-1 ciphertext `(out_bx, out_ax)` at the same level.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<AnyMemRef, "input in_rx (c2) buffer", [MemRead]>:$in_rx,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` $in_rx `,` `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)`
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_RelinearizeRescaleOp : CheddarGPU_Op<"relinearize_rescale", [AttrSizedOperandSegments]> {
  let summary = "Combined Relinearize and Rescale";
  let description = [{
    Relinearizes degree-2 ciphertext `(in_bx, in_ax, in_rx)` at `level` and rescales down to `level - 1`.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<AnyMemRef, "input in_rx (c2) buffer", [MemRead]>:$in_rx,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` $in_rx `,` `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)`
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_HRotOp : CheddarGPU_Op<"hrot", [AttrSizedOperandSegments]> {
  let summary = "Homomorphic slot rotation";
  let description = [{
    Rotates ciphertext `(in_bx, in_ax)` by `rot_dist` (or corresponding `galois` element) using
    rotation evaluation key `(evk_bx, evk_ax)`.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    I64Attr:$rot_dist,
    OptionalAttr<I64Attr>:$galois,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)`
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_HConjOp : CheddarGPU_Op<"hconj", [AttrSizedOperandSegments]> {
  let summary = "Homomorphic complex conjugation";
  let description = [{
    Conjugates ciphertext `(in_bx, in_ax)` using conjugation evaluation key `(evk_bx, evk_ax)`.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)`
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_HRotAddOp : CheddarGPU_Op<"hrot_add", [AttrSizedOperandSegments]> {
  let summary = "Homomorphic rotation followed by ciphertext addition";
  let description = [{
    Rotates ciphertext `(in1_bx, in1_ax)` and adds `(in2_bx, in2_ax)` into `(out_bx, out_ax)`.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input 1 in1_bx buffer", [MemRead]>:$in1_bx,
    Arg<AnyMemRef, "input 1 in1_ax buffer", [MemRead]>:$in1_ax,
    Arg<AnyMemRef, "input 2 in2_bx buffer", [MemRead]>:$in2_bx,
    Arg<AnyMemRef, "input 2 in2_ax buffer", [MemRead]>:$in2_ax,
    Arg<Variadic<AnyMemRef>, "evaluation key bx digits", [MemRead]>:$evk_bx,
    Arg<Variadic<AnyMemRef>, "evaluation key ax digits", [MemRead]>:$evk_ax,
    I64Attr:$level,
    I64Attr:$rot_dist,
    OptionalAttr<I64Attr>:$galois,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in1_bx `,` $in1_ax `,` $in2_bx `,` $in2_ax `,` `evk_bx` `(` $evk_bx `)` `,` `evk_ax` `(` $evk_ax `)`
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_ModUpToMaxOp : CheddarGPU_Op<"mod_up_to_max"> {
  let summary = "Broadcast and reduce polynomial limb across primes without centering";
  let description = [{
    Mirrors Cheddar's `ModUpToMax1` in `ElementWise.cu:580-592`.
    Computes `dst[l, x] = src[0, x] % primes[l]` pointwise for each limb `l in [0, dst_limbs)`.
    Operates in coefficient (`coeff`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    Arg<AnyMemRef, "source polynomial buffer", [MemRead]>:$in,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<DenseI64ArrayAttr>:$out_limb_range,
    OptionalAttr<I64Attr>:$prime_offset,
    OptionalAttr<I64Attr>:$num_aux
  );

  let assemblyFormat = "$out `,` $in prop-dict attr-dict `:` type($out) `,` type($in)";

  let hasVerifier = 1;
}

def CheddarGPU_ZeroPadOp : CheddarGPU_Op<"zero_pad"> {
  let summary = "Zero-fill limb range [start, end) of a buffer";
  let description = [{
    Zero-fills limbs in `[start, end)` of the destination buffer.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination buffer", [MemWrite]>:$out,
    DenseI64ArrayAttr:$limb_range
  );

  let assemblyFormat = "$out prop-dict attr-dict `:` type($out)";

  let hasVerifier = 1;
}

def CheddarGPU_HoistBSFusedOp : CheddarGPU_Op<"hoist_bs_fused", [AttrSizedOperandSegments]> {
  let summary = "Fused baby-step key multiplication, accumulation, and Galois automorphism";
  let description = [{
    Mirrors Cheddar's `BSFusedKernel` in `Hoist.cu:20-78`.
    Computes `R` rotated baby-step ciphertexts from mod-up digits and EVK slab.
  }];

  let arguments = (ins
    Arg<Variadic<AnyMemRef>, "destination bx polynomial buffers", [MemWrite]>:$out_bx,
    Arg<Variadic<AnyMemRef>, "destination ax polynomial buffers", [MemWrite]>:$out_ax,
    Arg<Variadic<AnyMemRef>, "mod-up digits of input ax", [MemRead]>:$mod_up,
    Arg<AnyMemRef, "EVK slab buffer", [MemRead]>:$evk_slab,
    Arg<AnyMemRef, "pseudo-modup input bx", [MemRead]>:$in_bx_pseudo_modup,
    DenseI64ArrayAttr:$galois_factors,
    DenseI64ArrayAttr:$evk_byte_offsets,
    DenseI64ArrayAttr:$evk_num_q,
    I64Attr:$num_q,
    I64Attr:$num_aux,
    I64Attr:$prime_offset
  );

  let assemblyFormat = [{
    `out_bx` `(` $out_bx `)` `,` `out_ax` `(` $out_ax `)` `,` `mod_up` `(` $mod_up `)` `,` $evk_slab `,` $in_bx_pseudo_modup
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_HoistGSFusedOp : CheddarGPU_Op<"hoist_gs_fused", [AttrSizedOperandSegments]> {
  let summary = "Fused giant-step plaintext multiplication and accumulation";
  let description = [{
    Mirrors Cheddar's `GSFusedKernel` in `Hoist.cu:80-118`.
    Computes `G` giant-step ciphertexts from `B` baby-step ciphertexts and plaintext constants slab.
  }];

  let arguments = (ins
    Arg<Variadic<AnyMemRef>, "destination bx polynomial buffers", [MemWrite]>:$out_bx,
    Arg<Variadic<AnyMemRef>, "destination ax polynomial buffers", [MemWrite]>:$out_ax,
    Arg<Variadic<AnyMemRef>, "input bx baby-step ciphertexts", [MemRead]>:$in_bx,
    Arg<Variadic<AnyMemRef>, "input ax baby-step ciphertexts", [MemRead]>:$in_ax,
    Arg<AnyMemRef, "plaintext constants slab buffer", [MemRead]>:$consts_slab,
    DenseI64ArrayAttr:$pt_word_offsets,
    I64Attr:$num_q,
    I64Attr:$num_aux,
    I64Attr:$prime_offset,
    I64Attr:$pt_limbs
  );

  let assemblyFormat = [{
    `out_bx` `(` $out_bx `)` `,` `out_ax` `(` $out_ax `)` `,` `in_bx` `(` $in_bx `)` `,` `in_ax` `(` $in_ax `)` `,` $consts_slab
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_EvalModOp : CheddarGPU_Op<"eval_mod"> {
  let summary = "Homomorphic modular reduction for CKKS bootstrapping";
  let description = [{
    Evaluates modular reduction polynomial on ciphertext `(in_bx, in_ax)` at level 21
    producing output ciphertext `(out_bx, out_ax)` at level 13.
    Uses multiplication key slice from `evk_slab`.
    Operates in evaluation (`eval`) domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<AnyMemRef, "evaluation key byte slab", [MemRead]>:$evk_slab,
    I64Attr:$relin_evk_byte_offset,
    I64Attr:$relin_evk_num_q,
    I64Attr:$relin_evk_beta,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<CheddarGPU_ParamsAttr>:$params
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` $evk_slab prop-dict attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_HoistPhaseOp : CheddarGPU_Op<"hoist_phase"> {
  let summary = "High-level hoisted linear transform phase for CKKS bootstrapping";
  let description = [{
    Represents one phase of CoeffToSlot or SlotToCoeff hoisted BSGS linear transform.
    Expanded by `--cheddar-gpu-expand-hoist` into fused BSGS kernels and giant-step reduction.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<AnyMemRef, "plaintext constants byte slab", [MemRead]>:$consts_slab,
    Arg<AnyMemRef, "evaluation key byte slab", [MemRead]>:$evk_slab,
    StrAttr:$kind,
    I64Attr:$phase,
    I64Attr:$num_slots,
    OptionalAttr<CheddarGPU_ParamsAttr>:$params
  );

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` $in_bx `,` $in_ax `,` $consts_slab `,` $evk_slab prop-dict attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CheddarGPU_BootOp : CheddarGPU_Op<"boot"> {
  let summary = "High-level bootstrapping operation for CKKS ciphertext";
  let description = [{
    Performs full bootstrapping on ciphertext `(in_bx, in_ax)` using `consts_slab`
    and `evk_slab`, producing refreshed ciphertext `(out_bx, out_ax)`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "destination out_bx buffer", [MemWrite]>:$out_bx,
    Arg<AnyMemRef, "destination out_ax buffer", [MemWrite]>:$out_ax,
    Arg<AnyMemRef, "input in_bx buffer", [MemRead]>:$in_bx,
    Arg<AnyMemRef, "input in_ax buffer", [MemRead]>:$in_ax,
    Arg<AnyMemRef, "plaintext constants byte slab", [MemRead]>:$consts_slab,
    Arg<AnyMemRef, "evaluation key byte slab", [MemRead]>:$evk_slab,
    I64Attr:$num_slots,
    CheddarGPU_FormAttr:$in_form,
    CheddarGPU_FormAttr:$out_form,
    OptionalAttr<CheddarGPU_ParamsAttr>:$params
  );

  let assemblyFormat = [{
    `ins` `(` $in_bx `,` $in_ax `,` $consts_slab `,` $evk_slab `:` type($in_bx) `,` type($in_ax) `,` type($consts_slab) `,` type($evk_slab) `)` `outs` `(` $out_bx `,` $out_ax `:` type($out_bx) `,` type($out_ax) `)` prop-dict attr-dict
  }];

  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
