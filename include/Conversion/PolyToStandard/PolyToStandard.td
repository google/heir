#ifndef INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
#define INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def PolyToStandard : Pass<"poly-to-standard", "ModuleOp"> {
  let summary = "Lower `poly` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `poly` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::heir::poly::PolyDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
