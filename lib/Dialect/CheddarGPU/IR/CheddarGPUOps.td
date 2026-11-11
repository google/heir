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

defvar CheddarGPU_BinaryForms = (ins
  CheddarGPU_FormAttr:$lhs_form,
  CheddarGPU_FormAttr:$rhs_form,
  CheddarGPU_FormAttr:$out_form
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

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
