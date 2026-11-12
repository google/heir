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

defvar CheddarGPU_MulConstBuilders = [
  OpBuilder<(ins "::mlir::Value":$out,
                 "::mlir::Value":$in,
                 "::mlir::Value":$cst,
                 "::mlir::heir::cheddar_gpu::FormAttr":$in_form,
                 "::mlir::heir::cheddar_gpu::FormAttr":$out_form,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$in_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$out_limb_range,
                 CArg<"::mlir::IntegerAttr", "{}">:$prime_offset), [{
      build($_builder, $_state, out, in, cst, in_form, out_form,
            in_limb_range, out_limb_range, prime_offset, /*num_aux=*/::mlir::IntegerAttr{});
  }]>,
  OpBuilder<(ins "::mlir::Value":$out,
                 "::mlir::Value":$in,
                 "::mlir::Value":$cst,
                 "::mlir::heir::cheddar_gpu::Form":$in_form,
                 "::mlir::heir::cheddar_gpu::Form":$out_form,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$in_limb_range,
                 CArg<"::mlir::DenseI64ArrayAttr", "{}">:$out_limb_range,
                 CArg<"::mlir::IntegerAttr", "{}">:$prime_offset), [{
      build($_builder, $_state, out, in, cst, in_form, out_form,
            in_limb_range, out_limb_range, prime_offset, /*num_aux=*/::mlir::IntegerAttr{});
  }]>
];

//===----------------------------------------------------------------------===//
// Operation Families
//===----------------------------------------------------------------------===//

// Binary element-wise operations on single polynomial buffers (add, sub, mul).
class CheddarGPU_BinaryOpBase<string mnemonic, list<Trait> traits = []>
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
    CheddarGPU_BinaryForms,
    CheddarGPU_BinaryLimbRanges,
    (ins OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux)
  );

  let builders = CheddarGPU_BinaryBuilders;

  let assemblyFormat = [{
    $out `,` $lhs `,` $rhs prop-dict attr-dict `:` type($out)
  }];
}

// Unary element-wise operations on single polynomial buffers (neg).
class CheddarGPU_UnaryOpBase<string mnemonic, dag extraAttrs = (ins),
                             list<OpBuilder> opBuilders = [], list<Trait> traits = []>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out", "in"]>,
        CheddarGPU_LimbRangesWithin
      ], traits)> {
  let arguments = !con(
    (ins
      Arg<CheddarGPU_PolyBuffer, "destination buffer", [MemWrite]>:$out,
      Arg<CheddarGPU_PolyBuffer, "input polynomial buffer", [MemRead]>:$in
    ),
    CheddarGPU_UnaryForms,
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
class CheddarGPU_ConstOpBase<
    string mnemonic,
    dag constArg = (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    dag extraAttrs = (ins),
    list<OpBuilder> opBuilders = [],
    list<Trait> traits = []>
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
    CheddarGPU_UnaryForms,
    CheddarGPU_UnaryLimbRanges,
    extraAttrs
  );

  let builders = opBuilders;
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

def CheddarGPU_MulOp : CheddarGPU_BinaryOpBase<"mul", [
    CheddarGPU_FormsAre<["lhs_form", "rhs_form", "out_form"], CheddarGPU_Form_Eval>]> {
  let summary = "Pointwise Montgomery modular multiplication";
  let description = [{
    Mirrors Cheddar's `kernel::Mult` and `ElementWiseHandler::Mult` in `ElementWise.cu`.
    Computes the Montgomery form multiplication of `lhs` and `rhs` mod `Q`.

    Form contract:
    - Polynomial multiplication in evaluation domain is pointwise Montgomery multiplication:
      `(a * R) * (b * R) * R^-1 = (a * b) * R mod Q`.
    - Operands must be in `eval` form (`lhs_form == rhs_form == out_form == eval`).
  }];
}

def CheddarGPU_MulConstOp : CheddarGPU_ConstOpBase<"mul_const",
    (ins Arg<CheddarGPU_LimbVector, "per-limb constant buffer", [MemRead]>:$cst),
    (ins
      OptionalAttr<I64Attr>:$prime_offset,
      OptionalAttr<CheddarGPU_NonNegI64Attr>:$num_aux),
    CheddarGPU_MulConstBuilders,
    [CheddarGPU_FormsAre<["in_form", "out_form"], CheddarGPU_Form_Eval>,
     CheddarGPU_PerLimbOf<"cst", "out">]> {
  let summary = "Pointwise Montgomery multiplication by per-limb constants";
  let description = [{
    Mirrors Cheddar's `ElementWiseHandler::MultConst` in `ElementWise.cu`.
    Computes the Montgomery form multiplication of `in` and `cst` mod `Q`
    limb-wise, where the limbs of `Q` are defined in `CheddarGPU_ParamsAttr`.

    Form contract:
    - Montgomery multiplication requires `eval` form (`in_form == out_form == eval`).
  }];
}

//===----------------------------------------------------------------------===//
// Imaginary Unit Operation
//===----------------------------------------------------------------------===//

def CheddarGPU_MulImaginaryUnitOp : CheddarGPU_ConstOpBase<"mul_imag_unit",
    (ins Arg<CheddarGPU_LimbVector, "per-limb imaginary unit table (psi^(N/2))", [MemRead]>:$i_unit),
    (ins), [],
    [CheddarGPU_FormsAre<["in_form", "out_form"], CheddarGPU_Form_Eval>,
     CheddarGPU_PerLimbOf<"i_unit", "out">]> {
  let summary = "Pointwise multiplication by imaginary unit in evaluation domain";
  let description = [{
    Mirrors Cheddar's `kernel::MultImaginaryUnit` in `ElementWise.cu`.
    In evaluation domain, multiplying by the imaginary unit corresponds to multiplying
    first half by `psi^(N/2)` and second half by `-psi^(N/2) = Q - psi^(N/2)`.

    Form contract:
    - Requires `eval` form (`in_form == out_form == eval`).
  }];
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
