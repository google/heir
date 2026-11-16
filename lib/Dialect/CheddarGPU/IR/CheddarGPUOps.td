#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_

include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.td"
include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.td"
include "lib/Dialect/CheddarGPU/IR/CheddarGPUConstraints.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Operation Class and Common Attribute/Builder Definitions
//===----------------------------------------------------------------------===//

class CheddarGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<CheddarGPU_Dialect, mnemonic, traits> {
  let hasVerifier = 0;
  let assemblyFormat = [{
    operands prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

defvar CheddarGPU_UnaryForms = (ins
  CheddarGPU_FormAttr:$in_form,
  CheddarGPU_FormAttr:$out_form
);

defvar CheddarGPU_BinaryForms = (ins
  CheddarGPU_FormAttr:$lhs_form,
  CheddarGPU_FormAttr:$rhs_form,
  CheddarGPU_FormAttr:$out_form
);

defvar CheddarGPU_UnaryLimbRanges = (ins
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$in_limb_range,
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range
);

defvar CheddarGPU_BinaryLimbRanges = (ins
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$lhs_limb_range,
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$rhs_limb_range,
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range
);

defvar CheddarGPU_AccumCommonAttrs = (ins
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range,
  OptionalAttr<I64Attr>:$prime_offset,
  OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux
);

defvar CheddarGPU_CPAccumAddCommonAttrs = (ins
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range,
  OptionalAttr<I64Attr>:$prime_offset,
  OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux
);

defvar CheddarGPU_PermuteAccumCommonAttrs = (ins
  DenseI64ArrayAttr:$galois_factors,
  OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range,
  OptionalAttr<I64Attr>:$prime_offset,
  OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux
);

defvar CheddarGPU_BinaryBuilders = [
  OpBuilder<(ins "::mlir::Value":$out,
                 "::mlir::Value":$lhs,
                 "::mlir::Value":$rhs,
                 "::mlir::heir::cheddar_gpu::FormAttr":$lhs_form,
                 "::mlir::heir::cheddar_gpu::FormAttr":$rhs_form,
                 "::mlir::heir::cheddar_gpu::FormAttr":$out_form,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/::mlir::IntegerAttr{});
  }]>,
  OpBuilder<(ins "::mlir::Value":$out,
                 "::mlir::Value":$lhs,
                 "::mlir::Value":$rhs,
                 "::mlir::heir::cheddar_gpu::Form":$lhs_form,
                 "::mlir::heir::cheddar_gpu::Form":$rhs_form,
                 "::mlir::heir::cheddar_gpu::Form":$out_form,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$lhs_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$rhs_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, lhs, rhs, lhs_form, rhs_form, out_form,
            lhs_limb_range, rhs_limb_range, out_limb_range, /*num_aux=*/::mlir::IntegerAttr{});
  }]>
];

defvar CheddarGPU_UnaryBuilders = [
  OpBuilder<(ins "::mlir::Value":$out,
                 "::mlir::Value":$in,
                 "::mlir::heir::cheddar_gpu::FormAttr":$in_form,
                 "::mlir::heir::cheddar_gpu::FormAttr":$out_form,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$in_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, in, in_form, out_form,
            in_limb_range, out_limb_range, /*num_aux=*/::mlir::IntegerAttr{});
  }]>,
  OpBuilder<(ins "::mlir::Value":$out,
                 "::mlir::Value":$in,
                 "::mlir::heir::cheddar_gpu::Form":$in_form,
                 "::mlir::heir::cheddar_gpu::Form":$out_form,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$in_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$out_limb_range), [{
      build($_builder, $_state, out, in, in_form, out_form,
            in_limb_range, out_limb_range, /*num_aux=*/::mlir::IntegerAttr{});
  }]>
];

//===----------------------------------------------------------------------===//
// Operation Families
//===----------------------------------------------------------------------===//

// Binary element-wise operations on single polynomial buffers (add, sub, mul).
// Ops that accept only one form set `hasForms = 0`; their forms are implied.
class CheddarGPU_BinaryOpBase<string mnemonic, list<Trait> traits = [],
                              bit hasForms = 1>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out", "lhs", "rhs"]>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination buffer", [MemWrite]>:$out,
      Arg<CheddarGPU_PolyBuffer, "left-hand side operand", [MemRead]>:$lhs,
      Arg<CheddarGPU_PolyBuffer, "right-hand side operand", [MemRead]>:$rhs
    ),
    !if(hasForms, CheddarGPU_BinaryForms, (ins)),
    CheddarGPU_BinaryLimbRanges,
    (ins OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux)
  );

  let builders = !if(hasForms, CheddarGPU_BinaryBuilders, []);

  let assemblyFormat = [{
    $out `,` $lhs `,` $rhs prop-dict attr-dict `:` type($out)
  }];
}

// Unary element-wise operations on single polynomial buffers (neg).
// Ops that accept only one form set `hasForms = 0`; their forms are implied.
class CheddarGPU_UnaryOpBase<string mnemonic, dag extraAttrs = (ins),
                             list<OpBuilder> opBuilders = [], list<Trait> traits = [],
                             bit hasForms = 1>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out", "in"]>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination buffer", [MemWrite]>:$out,
      Arg<CheddarGPU_PolyBuffer, "input polynomial buffer", [MemRead]>:$in
    ),
    !if(hasForms, CheddarGPU_UnaryForms, (ins)),
    extraAttrs,
    CheddarGPU_UnaryLimbRanges,
    (ins OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux)
  );

  let builders = opBuilders;
  let assemblyFormat = [{
    $out `,` $in prop-dict attr-dict `:` type($out)
  }];
}

// Element-wise operations with per-limb constants (add_const, sub_const,
// sub_opposite_const, mul_const, mul_imag_unit).
// Ops that accept only one form set `hasForms = 0`; their forms are implied.
class CheddarGPU_ConstOpBase<
    string mnemonic,
    dag constArg = (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    dag extraAttrs = (ins),
    list<OpBuilder> opBuilders = [],
    list<Trait> traits = [],
    bit hasForms = 1>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out", "in"]>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination buffer", [MemWrite]>:$out,
      Arg<CheddarGPU_PolyBuffer, "input polynomial buffer", [MemRead]>:$in
    ),
    constArg,
    !if(hasForms, CheddarGPU_UnaryForms, (ins)),
    CheddarGPU_UnaryLimbRanges,
    extraAttrs
  );

  let builders = opBuilders;
}

// Binary operations on pairs of polynomial buffers (add_2, sub_2).
class CheddarGPU_PairBinaryOpBase<string mnemonic, list<Trait> traits = []>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out_bx", "out_ax", "lhs_bx", "lhs_ax", "rhs_bx", "rhs_ax"]>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination bx buffer", [MemWrite]>:$out_bx,
      Arg<CheddarGPU_PolyBuffer, "destination ax buffer", [MemWrite]>:$out_ax,
      Arg<CheddarGPU_PolyBuffer, "lhs bx buffer", [MemRead]>:$lhs_bx,
      Arg<CheddarGPU_PolyBuffer, "lhs ax buffer", [MemRead]>:$lhs_ax,
      Arg<CheddarGPU_PolyBuffer, "rhs bx buffer", [MemRead]>:$rhs_bx,
      Arg<CheddarGPU_PolyBuffer, "rhs ax buffer", [MemRead]>:$rhs_ax
    ),
    CheddarGPU_BinaryForms,
    CheddarGPU_BinaryLimbRanges,
    (ins OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux)
  );

  let assemblyFormat = [{
    `ins` `(` $lhs_bx `,` $lhs_ax `,` $rhs_bx `,` $rhs_ax `)`
    `outs` `(` $out_bx `,` $out_ax `)`
    prop-dict attr-dict `:` type($out_bx)
  }];
}

// Unary operations on pairs of polynomial buffers (neg_2).
// Ops that accept only one form set `hasForms = 0`; their forms are implied.
class CheddarGPU_PairUnaryOpBase<string mnemonic, dag extraAttrs = (ins), list<Trait> traits = [],
                                 bit hasForms = 1>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out_bx", "out_ax", "in_bx", "in_ax"]>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination bx buffer", [MemWrite]>:$out_bx,
      Arg<CheddarGPU_PolyBuffer, "destination ax buffer", [MemWrite]>:$out_ax,
      Arg<CheddarGPU_PolyBuffer, "in bx buffer", [MemRead]>:$in_bx,
      Arg<CheddarGPU_PolyBuffer, "in ax buffer", [MemRead]>:$in_ax
    ),
    !if(hasForms, CheddarGPU_UnaryForms, (ins)),
    extraAttrs,
    CheddarGPU_UnaryLimbRanges,
    (ins OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux)
  );

  let assemblyFormat = [{
    `ins` `(` $in_bx `,` $in_ax `)`
    `outs` `(` $out_bx `,` $out_ax `)`
    prop-dict attr-dict `:` type($out_bx)
  }];
}

// Ciphertext tensor product and squaring operations (tensor, tensor_square).
class CheddarGPU_TensorOpBase<string mnemonic, dag srcOperands,
                              list<string> matchedTypes, list<Trait> traits = []>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<!listconcat(["out_bx", "out_ax", "out_rx"], matchedTypes)>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination bx buffer", [MemWrite]>:$out_bx,
      Arg<CheddarGPU_PolyBuffer, "destination ax buffer", [MemWrite]>:$out_ax,
      Arg<CheddarGPU_PolyBuffer, "destination rx buffer", [MemWrite]>:$out_rx
    ),
    srcOperands,
    (ins OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range)
  );
}

// Accumulation operations base classes.
class CheddarGPU_AccumOpBase<string mnemonic, dag inOperands, dag accumAttrs,
                             list<Trait> traits = []>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AttrSizedOperandSegments,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins Arg<CheddarGPU_PolyBuffer, "destination buffer", [MemWrite]>:$out),
    inOperands,
    accumAttrs
  );
}

class CheddarGPU_PairAccumOpBase<string mnemonic, dag inOperands, dag accumAttrs,
                                 list<Trait> traits = []>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AttrSizedOperandSegments,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination bx buffer", [MemWrite]>:$out_bx,
      Arg<CheddarGPU_PolyBuffer, "destination ax buffer", [MemWrite]>:$out_ax
    ),
    inOperands,
    accumAttrs
  );
}

//===----------------------------------------------------------------------===//
// Memory and Data Movement Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_CopyOp : CheddarGPU_Op<"copy", [
    AllElementTypesMatch<["out", "in"]>,
    CheddarGPU_SameDegree<["out", "in"]>,
    CheddarGPU_LimbRangesWithin,
    CheddarGPU_InOutCountsMatch
]> {
  let summary = "Pointwise copy of a polynomial buffer or slice";
  let description = [{
    Copies elements from `in` to `out`.
    Supports optional `in_limb_range` and `out_limb_range` [start, count].
  }];

  let arguments = (ins
    Arg<CheddarGPU_PolyBuffer, "destination buffer", [MemWrite]>:$out,
    Arg<CheddarGPU_PolyBuffer, "source buffer", [MemRead]>:$in,
    OptionalAttr<CheddarGPU_LimbRangeAttr>:$in_limb_range,
    OptionalAttr<CheddarGPU_LimbRangeAttr>:$out_limb_range
  );
}

//===----------------------------------------------------------------------===//
// Binary and Unary Element-Wise Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_AddOp : CheddarGPU_BinaryOpBase<"add", [
    CheddarGPU_FormsMatch<["lhs_form", "rhs_form", "out_form"]>]> {
  let summary = "Pointwise modular addition of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `kernel::Sum` and `ElementWiseHandler::Add` in `ElementWise.cu`.
    Computes `out = lhs + rhs mod Q` limb-wise, where the limbs of `Q` are defined
    in `CheddarGPU_ParamsAttr`.

    Form contract:
    - Modular addition is valid in both `coeff` and `eval` domain.
    - Both operands must have matching form (`lhs_form == rhs_form == out_form`).
  }];
}

def CheddarGPU_SubOp : CheddarGPU_BinaryOpBase<"sub", [
    CheddarGPU_FormsMatch<["lhs_form", "rhs_form", "out_form"]>]> {
  let summary = "Pointwise modular subtraction of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `kernel::Sub` and `ElementWiseHandler::Sub` in `ElementWise.cu`.
    Computes `out = lhs - rhs mod Q` limb-wise, where the limbs of `Q` are defined
    in `CheddarGPU_ParamsAttr`.

    Form contract:
    - Modular subtraction is valid in both `coeff` and `eval` domain.
    - Both operands must have matching form (`lhs_form == rhs_form == out_form`).
  }];
}

def CheddarGPU_NegOp : CheddarGPU_UnaryOpBase<"neg", (ins), CheddarGPU_UnaryBuilders, [
    CheddarGPU_FormsMatch<["in_form", "out_form"]>]> {
  let summary = "Pointwise modular negation of a polynomial buffer";
  let description = [{
    Mirrors Cheddar's `kernel::Neg` and `ElementWiseHandler::Neg` in `ElementWise.cu`.
    Computes `out = -in mod Q` limb-wise, where the limbs of `Q` are defined
    in `CheddarGPU_ParamsAttr`.

    Form contract:
    - Valid in both `coeff` and `eval` domain.
    - Preserves representation form (`in_form == out_form`).
  }];
}

def CheddarGPU_AddConstOp : CheddarGPU_ConstOpBase<"add_const",
    (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    (ins), [],
    [CheddarGPU_FormsMatch<["in_form", "out_form"]>, CheddarGPU_PerLimbOf<"cst", "out">]> {
  let summary = "Pointwise addition of per-limb constants to a polynomial buffer";
  let description = [{
    Mirrors Cheddar's `kernel::AddConst` and `ElementWiseHandler::AddConst` in `ElementWise.cu`.
    Computes `out[l][i] = in[l][i] + cst[l] mod Q_l`, where `Q_l` is the `l`-th limb
    of `Q` from `CheddarGPU_ParamsAttr`.

    Form contract:
    - Valid in both `coeff` and `eval` domain (`in_form == out_form`).
  }];
}

def CheddarGPU_SubConstOp : CheddarGPU_ConstOpBase<"sub_const",
    (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    (ins), [],
    [CheddarGPU_FormsMatch<["in_form", "out_form"]>, CheddarGPU_PerLimbOf<"cst", "out">]> {
  let summary = "Pointwise subtraction of per-limb constants from a polynomial buffer";
  let description = [{
    Mirrors Cheddar's `kernel::SubConst` and `ElementWiseHandler::SubConst` in `ElementWise.cu`.
    Computes `out[l][i] = in[l][i] - cst[l] mod Q_l`, where `Q_l` is the `l`-th limb
    of `Q` from `CheddarGPU_ParamsAttr`.

    Form contract:
    - Valid in both `coeff` and `eval` domain (`in_form == out_form`).
  }];
}

def CheddarGPU_SubOppositeConstOp : CheddarGPU_ConstOpBase<"sub_opposite_const",
    (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    (ins), [],
    [CheddarGPU_FormsMatch<["in_form", "out_form"]>, CheddarGPU_PerLimbOf<"cst", "out">]> {
  let summary = "Pointwise subtraction of polynomial from per-limb constants";
  let description = [{
    Mirrors Cheddar's `kernel::SubOppositeConst` and `ElementWiseHandler::SubOppositeConst` in `ElementWise.cu`.
    Computes `out[l][i] = cst[l] - in[l][i] mod Q_l`, where `Q_l` is the `l`-th limb
    of `Q` from `CheddarGPU_ParamsAttr`.

    Form contract:
    - Valid in both `coeff` and `eval` domain (`in_form == out_form`).
  }];
}

def CheddarGPU_MulOp : CheddarGPU_BinaryOpBase<"mul", hasForms = 0> {
  let summary = "Pointwise Montgomery modular multiplication";
  let description = [{
    Mirrors Cheddar's `kernel::Mult` and `ElementWiseHandler::Mult` in `ElementWise.cu`.
    Computes the Montgomery form multiplication of `lhs` and `rhs` mod `Q`.

    Form contract:
    - Polynomial multiplication in evaluation domain is pointwise Montgomery multiplication:
      `(a * R) * (b * R) * R^-1 = (a * b) * R mod Q`.
    - All operands are in `eval` form.
  }];
}

def CheddarGPU_MulConstOp : CheddarGPU_ConstOpBase<"mul_const",
    (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    (ins
      OptionalAttr<I64Attr>:$prime_offset,
      OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux),
    [],
    [CheddarGPU_PerLimbOf<"cst", "out">],
    hasForms = 0> {
  let summary = "Pointwise Montgomery multiplication by per-limb constants";
  let description = [{
    Mirrors Cheddar's `ElementWiseHandler::MultConst` in `ElementWise.cu`.
    Computes the Montgomery form multiplication of `in` and `cst` mod `Q`
    limb-wise, where the limbs of `Q` are defined in `CheddarGPU_ParamsAttr`.

    Form contract:
    - `in` and `out` are in `eval` form.
  }];
}

//===----------------------------------------------------------------------===//
// Imaginary Unit Operation
//===----------------------------------------------------------------------===//

def CheddarGPU_MulImaginaryUnitOp : CheddarGPU_ConstOpBase<"mul_imag_unit",
    (ins Arg<CheddarGPU_LimbVector, "per-limb imaginary unit table (psi^(N/2))", [MemRead]>:$i_unit),
    (ins), [],
    [CheddarGPU_PerLimbOf<"i_unit", "out">],
    hasForms = 0> {
  let summary = "Pointwise multiplication by imaginary unit in evaluation domain";
  let description = [{
    Mirrors Cheddar's `kernel::MultImaginaryUnit` in `ElementWise.cu`.
    In evaluation domain, multiplying by the imaginary unit corresponds to multiplying
    first half by `psi^(N/2)` and second half by `-psi^(N/2) = Q - psi^(N/2)`.

    Form contract:
    - `in` and `out` are in `eval` form.
  }];
}

//===----------------------------------------------------------------------===//
// Pair Binary and Unary Element-Wise Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_Add2Op : CheddarGPU_PairBinaryOpBase<"add_2", [
    CheddarGPU_FormsMatch<["lhs_form", "rhs_form", "out_form"]>]> {
  let summary = "Pointwise modular addition of two pairs of polynomial buffers";
  let description = [{
    Mirrors Cheddar's `ElementWiseHandler::Add` with two polynomials, which launches
    `kernel::Sum<word, 2>` in `ElementWise.cu`.
    Computes `out_bx = lhs_bx + rhs_bx mod Q` and `out_ax = lhs_ax + rhs_ax mod Q`
    limb-wise, where the limbs of `Q` are defined in `CheddarGPU_ParamsAttr`.
  }];
}

def CheddarGPU_Sub2Op : CheddarGPU_PairBinaryOpBase<"sub_2", [
    CheddarGPU_FormsMatch<["lhs_form", "rhs_form", "out_form"]>]> {
  let summary = "Pointwise modular subtraction of two pairs of polynomial buffers";
  let description = [{
    Mirrors Cheddar's `ElementWiseHandler::Sub` with two polynomials, which launches
    `kernel::Sub<word, 2>` in `ElementWise.cu`.
    Computes `out_bx = lhs_bx - rhs_bx mod Q` and `out_ax = lhs_ax - rhs_ax mod Q`
    limb-wise, where the limbs of `Q` are defined in `CheddarGPU_ParamsAttr`.
  }];
}

def CheddarGPU_Neg2Op : CheddarGPU_PairUnaryOpBase<"neg_2", (ins), [
    CheddarGPU_FormsMatch<["in_form", "out_form"]>]> {
  let summary = "Pointwise modular negation of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `ElementWiseHandler::Neg` with two polynomials, which launches
    `kernel::Neg<word, 2>` in `ElementWise.cu`.
    Computes `out_bx = -in_bx mod Q` and `out_ax = -in_ax mod Q` limb-wise, where
    the limbs of `Q` are defined in `CheddarGPU_ParamsAttr`.
  }];
}

//===----------------------------------------------------------------------===//
// Ciphertext Tensor Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_TensorOp : CheddarGPU_TensorOpBase<"tensor",
    (ins
      Arg<CheddarGPU_PolyBuffer, "src1 bx buffer", [MemRead]>:$src1_bx,
      Arg<CheddarGPU_PolyBuffer, "src1 ax buffer", [MemRead]>:$src1_ax,
      Arg<CheddarGPU_PolyBuffer, "src2 bx buffer", [MemRead]>:$src2_bx,
      Arg<CheddarGPU_PolyBuffer, "src2 ax buffer", [MemRead]>:$src2_ax
    ),
    ["src1_bx", "src1_ax", "src2_bx", "src2_ax"]> {
  let summary = "Ciphertext tensor product: (b1*b2, b1*a2 + a1*b2, a1*a2)";
  let description = [{
    Mirrors Cheddar's `kernel::Tensor` and `ElementWiseHandler::Tensor` in `ElementWise.cu`.
    Multiplies two degree-1 ciphertexts (b1, a1) and (b2, a2) into a degree-2 ciphertext
    (b1*b2, b1*a2 + a1*b2, a1*a2) using Karatsuba modular multiplication.

    Form contract:
    - All operands are in `eval` form.
  }];

  let assemblyFormat = [{
    `ins` `(` $src1_bx `,` $src1_ax `,` $src2_bx `,` $src2_ax `)`
    `outs` `(` $out_bx `,` $out_ax `,` $out_rx `)`
    prop-dict attr-dict `:` type($out_bx)
  }];
}

def CheddarGPU_TensorSquareOp : CheddarGPU_TensorOpBase<"tensor_square",
    (ins
      Arg<CheddarGPU_PolyBuffer, "src bx buffer", [MemRead]>:$src_bx,
      Arg<CheddarGPU_PolyBuffer, "src ax buffer", [MemRead]>:$src_ax
    ),
    ["src_bx", "src_ax"]> {
  let summary = "Ciphertext tensor squaring: (b1^2, 2*b1*a1, a1^2)";
  let description = [{
    Mirrors Cheddar's `kernel::TensorSquare` in `ElementWise.cu`.
    Squares degree-1 ciphertext (b1, a1) into degree-2 ciphertext (b1^2, 2*b1*a1, a1^2).

    Form contract:
    - All operands are in `eval` form.
  }];

  let assemblyFormat = [{
    `ins` `(` $src_bx `,` $src_ax `)`
    `outs` `(` $out_bx `,` $out_ax `,` $out_rx `)`
    prop-dict attr-dict `:` type($out_bx)
  }];
}

//===----------------------------------------------------------------------===//
// Accumulation Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_CPAccumOp : CheddarGPU_AccumOpBase<"cp_accum",
    (ins
      Variadic<CheddarGPU_PolyBuffer>:$ct_srcs,
      Variadic<CheddarGPU_PolyOrLimbBuffer>:$common_srcs
    ),
    CheddarGPU_AccumCommonAttrs,
    [
      CheddarGPU_SameCount<["ct_srcs", "common_srcs"]>,
      CheddarGPU_NonEmpty<"ct_srcs">,
      CheddarGPU_EachSameRingAs<"ct_srcs", "out">,
      CheddarGPU_Each2DTypeIs<"common_srcs", "out">,
      CheddarGPU_EachCoversOutCount<"ct_srcs", "out">,
      CheddarGPU_EachCoversOutCount<"common_srcs", "out", 1>
    ]> {
  let summary = "Accumulate Montgomery products: dst = sum_k (common_srcs[k] * ct_srcs[k])";
  let description = [{
    Mirrors Cheddar's `kernel::CPAccum` and `ElementWiseHandler::CPAccumWorker<false/true>`
    in `ElementWise.cu`.
    Computes `out = sum_k MultMontgomery(ct_srcs[k], common_srcs[k])`.
    `common_srcs` can be polynomial buffers (`memref<LxNxiW>`) or per-limb constants (`memref<LxiW>`).

    Form contract:
    - Montgomery pointwise multiplication requires `eval` form for all operands.
  }];

  let assemblyFormat = [{
    $out `,` `ct_srcs` `(` $ct_srcs `)` `,` `common_srcs` `(` $common_srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

def CheddarGPU_CPAccum2Op : CheddarGPU_PairAccumOpBase<"cp_accum_2",
    (ins
      Variadic<CheddarGPU_PolyBuffer>:$ct_bx_srcs,
      Variadic<CheddarGPU_PolyBuffer>:$ct_ax_srcs,
      Variadic<CheddarGPU_PolyOrLimbBuffer>:$common_srcs
    ),
    CheddarGPU_AccumCommonAttrs,
    [
      AllTypesMatch<["out_bx", "out_ax"]>,
      CheddarGPU_SameCount<["ct_bx_srcs", "ct_ax_srcs", "common_srcs"]>,
      CheddarGPU_NonEmpty<"ct_bx_srcs">,
      CheddarGPU_EachSameRingAs<"ct_bx_srcs", "out_bx">,
      CheddarGPU_EachSameRingAs<"ct_ax_srcs", "out_ax">,
      CheddarGPU_Each2DTypeIs<"common_srcs", "out_bx">,
      CheddarGPU_EachCoversOutCount<"ct_bx_srcs", "out_bx">,
      CheddarGPU_EachCoversOutCount<"ct_ax_srcs", "out_ax">,
      CheddarGPU_EachCoversOutCount<"common_srcs", "out_bx", 1>
    ]> {
  let summary = "Accumulate Montgomery products for two polynomials: {out_bx, out_ax} = sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])";
  let description = [{
    Mirrors Cheddar's `CPAccum<word,2,...>` / `PAccum` in `Context.cpp` / `ElementWise.cu`.
    Computes `{out_bx, out_ax} = sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])` over `k < K` (K <= 4),
    reading each `common[k]` (mod-up digit) once for both outputs.
  }];

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` `ct_bx_srcs` `(` $ct_bx_srcs `)` `,`
    `ct_ax_srcs` `(` $ct_ax_srcs `)` `,` `common_srcs` `(` $common_srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

def CheddarGPU_CPAccumAddOp : CheddarGPU_AccumOpBase<"cp_accum_add",
    (ins
      Arg<CheddarGPU_PolyBuffer, "initial accumulator buffer", [MemRead]>:$acc,
      Variadic<CheddarGPU_PolyBuffer>:$ct_srcs,
      Variadic<CheddarGPU_PolyOrLimbBuffer>:$common_srcs
    ),
    CheddarGPU_CPAccumAddCommonAttrs,
    [
      AllTypesMatch<["out", "acc"]>,
      CheddarGPU_SameCount<["ct_srcs", "common_srcs"]>,
      CheddarGPU_NonEmpty<"ct_srcs">,
      CheddarGPU_EachSameRingAs<"ct_srcs", "out">,
      CheddarGPU_Each2DTypeIs<"common_srcs", "out">,
      CheddarGPU_EachCoversOutCount<"ct_srcs", "out">,
      CheddarGPU_EachCoversOutCount<"common_srcs", "out", 1>
    ]> {
  let summary = "Accumulate Montgomery products with base: dst = acc + sum_k (common_srcs[k] * ct_srcs[k])";
  let description = [{
    Mirrors Cheddar's `kernel::CPAccumAdd` in `ElementWise.cu`.
    Computes `out = acc + sum_k MultMontgomery(ct_srcs[k], common_srcs[k])`.

    Form contract:
    - Montgomery pointwise multiplication requires `eval` form for all operands.
  }];

  let assemblyFormat = [{
    $out `,` `acc` `(` $acc `)` `,` `ct_srcs` `(` $ct_srcs `)` `,`
    `common_srcs` `(` $common_srcs `)` prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def CheddarGPU_CPAccumAdd2Op : CheddarGPU_PairAccumOpBase<"cp_accum_add_2",
    (ins
      Arg<CheddarGPU_PolyBuffer, "initial accumulator bx buffer", [MemRead]>:$acc_bx,
      Arg<CheddarGPU_PolyBuffer, "initial accumulator ax buffer", [MemRead]>:$acc_ax,
      Variadic<CheddarGPU_PolyBuffer>:$ct_bx_srcs,
      Variadic<CheddarGPU_PolyBuffer>:$ct_ax_srcs,
      Variadic<CheddarGPU_PolyOrLimbBuffer>:$common_srcs
    ),
    CheddarGPU_CPAccumAddCommonAttrs,
    [
      AllTypesMatch<["out_bx", "out_ax", "acc_bx", "acc_ax"]>,
      CheddarGPU_SameCount<["ct_bx_srcs", "ct_ax_srcs", "common_srcs"]>,
      CheddarGPU_NonEmpty<"ct_bx_srcs">,
      CheddarGPU_EachSameRingAs<"ct_bx_srcs", "out_bx">,
      CheddarGPU_EachSameRingAs<"ct_ax_srcs", "out_ax">,
      CheddarGPU_Each2DTypeIs<"common_srcs", "out_bx">,
      CheddarGPU_EachCoversOutCount<"ct_bx_srcs", "out_bx">,
      CheddarGPU_EachCoversOutCount<"ct_ax_srcs", "out_ax">,
      CheddarGPU_EachCoversOutCount<"common_srcs", "out_bx", 1>
    ]> {
  let summary = "Accumulate Montgomery products with base for two polynomials: {out_bx, out_ax} = {acc_bx, acc_ax} + sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])";
  let description = [{
    Mirrors Cheddar's `CPAccumAdd<L,2>` / `CAccum` in `Context.cpp` / `ElementWise.cu`.
    Computes `{out_bx, out_ax} = {acc_bx, acc_ax} + sum_k MontMul({ct_bx[k], ct_ax[k]}, common[k])`.
  }];

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` `acc` `(` $acc_bx `,` $acc_ax `)` `,`
    `ct_bx_srcs` `(` $ct_bx_srcs `)` `,` `ct_ax_srcs` `(` $ct_ax_srcs `)` `,`
    `common_srcs` `(` $common_srcs `)` prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Permutation and Automorphism Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_PermuteOp : CheddarGPU_UnaryOpBase<"permute",
    (ins OptionalAttr<I64Attr>:$galois,
         UnitAttr:$conj), [],
    [
      CheddarGPU_ExactlyOneOf<"galois", "conj">
    ],
    hasForms = 0> {
  let summary = "Galois automorphism permutation in eval form";
  let description = [{
    Mirrors Cheddar's `kernel::Permute` and `ElementWiseHandler::Permute` in `ElementWise.cu`
    to compute a Galois automorphism on a polynomial.

    Computes:

      `e = brev_{logN+1}(x) + 1`
      `src = brev_{logN+1}(e * g - 1)`

    where `g` is either the Galois factor (`galois` attr, `5^k mod 2N`) or complex conjugation
    `2N - 1` (`conj` unit attr).

    Form contract:
    - Requires `eval` form for ciphertext operands.
    - in-place permutation is not supported.
  }];
}

def CheddarGPU_Permute2Op : CheddarGPU_PairUnaryOpBase<"permute_2",
    (ins OptionalAttr<I64Attr>:$galois,
         UnitAttr:$conj),
    [
      CheddarGPU_ExactlyOneOf<"galois", "conj">
    ],
    hasForms = 0> {
  let summary = "Galois automorphism permutation for two polynomials";
  let description = [{
    Same as `cheddar_gpu.permute` with two polynomials.

    Computes eval-form Galois automorphism on both bx and ax simultaneously.

    Note in-place permutation is not supported, so all operand buffers
    must be distinct.
  }];
}

def CheddarGPU_PermuteAccumOp : CheddarGPU_AccumOpBase<"permute_accum",
    (ins
      Optional<CheddarGPU_PolyBuffer>:$acc,
      Variadic<CheddarGPU_PolyBuffer>:$srcs
    ),
    CheddarGPU_PermuteAccumCommonAttrs,
    [
      OptionalTypesMatchWith<"accumulator matches output type", "out", "acc", "$_self">,
      CheddarGPU_SameCount<["srcs", "galois_factors"]>,
      CheddarGPU_NonEmpty<"srcs">,
      CheddarGPU_EachTypeIs<"srcs", "out">
    ]> {
  let summary = "Accumulate multiple Galois automorphism permutations";
  let description = [{
    Mirrors Cheddar's `kernel::PermuteAccum` and `kernel::PermuteAccumAdd` in `ElementWise.cu`.
    Computes `out = (acc +) sum_k permute(srcs[k], galois_factors[k])`.

    Form contract:
    - Requires `eval` form for all operands.
  }];

  let assemblyFormat = [{
    $out `,` (`acc` `(` $acc^ `)` `,`)? `srcs` `(` $srcs `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

def CheddarGPU_PermuteAccum2Op : CheddarGPU_PairAccumOpBase<"permute_accum_2",
    (ins
      Optional<CheddarGPU_PolyBuffer>:$acc_bx,
      Optional<CheddarGPU_PolyBuffer>:$acc_ax,
      Variadic<CheddarGPU_PolyBuffer>:$srcs_bx,
      Variadic<CheddarGPU_PolyBuffer>:$srcs_ax
    ),
    CheddarGPU_PermuteAccumCommonAttrs,
    [
      AllTypesMatch<["out_bx", "out_ax"]>,
      CheddarGPU_BothOrNeither<"acc_bx", "acc_ax">,
      OptionalTypesMatchWith<"acc_bx matches out_bx", "out_bx", "acc_bx", "$_self">,
      OptionalTypesMatchWith<"acc_ax matches out_ax", "out_ax", "acc_ax", "$_self">,
      CheddarGPU_SameCount<["srcs_bx", "srcs_ax", "galois_factors"]>,
      CheddarGPU_NonEmpty<"srcs_bx">,
      CheddarGPU_EachTypeIs<"srcs_bx", "out_bx">,
      CheddarGPU_EachTypeIs<"srcs_ax", "out_ax">
    ]> {
  let summary = "Accumulate multiple Galois automorphism permutations for two polynomials";
  let description = [{
    Mirrors Cheddar's `PermuteAccum` for 2 polynomials in `ElementWise.cu`.
    Computes `{out_bx, out_ax} = ({acc_bx, acc_ax} +) sum_k permute({srcs_bx[k], srcs_ax[k]}, galois_factors[k])`.
  }];

  let assemblyFormat = [{
    $out_bx `,` $out_ax `,` (`acc` `(` $acc_bx^ `,` $acc_ax `)` `,`)?
    `srcs_bx` `(` $srcs_bx `)` `,` `srcs_ax` `(` $srcs_ax `)` prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
