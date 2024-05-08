#ifndef LIB_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_TD_
#define LIB_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def PolynomialToStandard : Pass<"polynomial-to-standard", "ModuleOp"> {
  let summary = "Lower `polynomial` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `polynomial` dialect to standard MLIR, a mixture of
    affine, tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::polynomial::PolynomialDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_CONVERSION_POLYNOMIALTOSTANDARD_POLYNOMIALTOSTANDARD_TD_
