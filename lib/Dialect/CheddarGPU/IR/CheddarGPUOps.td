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
// CheddarGPU Operation Syntax Shapes
//===----------------------------------------------------------------------===//
//
// Every operation in the `cheddar_gpu` dialect adheres to declarative assembly
// format shapes:
//
// (a) Baseline:
//     `operands prop-dict attr-dict `:` functional-type(operands, results)`
//     Inherited directly from `CheddarGPU_Op`. Used for single-buffer elementwise
//     arithmetic (add, sub, neg, add_const, sub_const, sub_opposite_const, mul,
//     mul_const, mul_imag_unit), and memory copying (copy).

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

//===----------------------------------------------------------------------===//
// Operation Families
//===----------------------------------------------------------------------===//

// Binary element-wise operations on single polynomial buffers (add, sub, mul).
class CheddarGPU_BinaryOpBase<string mnemonic, list<Trait> traits = []>
    : CheddarGPU_Op<mnemonic, !listconcat([
        AllTypesMatch<["out", "lhs", "rhs"]>,
        CheddarGPU_LimbRangesWithinDest
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
}

//===----------------------------------------------------------------------===//
// Binary and Unary Element-Wise Operations
//===----------------------------------------------------------------------===//

def CheddarGPU_AddOp : CheddarGPU_BinaryOpBase<"add", [
    CheddarGPU_FormsMatch<["lhs_form", "rhs_form", "out_form"]>]> {
  let summary = "Pointwise modular addition of two polynomial buffers";
  let description = [{
    Mirrors Cheddar's `kernel::Sum` and `ElementWiseHandler::Add` in `ElementWise.cu`.
    Computes `out = lhs + rhs mod q` pointwise for each limb.

    Form contract:
    - Modular addition is valid in both `coeff` and `eval` domain.
    - Both operands must have matching form (`lhs_form == rhs_form == out_form`).
  }];
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_OPS_TD_
