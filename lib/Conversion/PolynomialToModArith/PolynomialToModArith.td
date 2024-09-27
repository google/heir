#ifndef LIB_CONVERSION_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_
#define LIB_CONVERSION_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_

include "mlir/Pass/PassBase.td"
include "lib/Dialect/ModArith/IR/ModArithDialect.td"

def PolynomialToModArith : Pass<"polynomial-to-mod-arith", "ModuleOp"> {
  let summary = "Lower `polynomial` to `mod-arith`.";

  let description = [{
    This pass lowers the `polynomial` dialect to `mod-arith`, with
    affine and tensor.
  }];
  let dependentDialects = [
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::polynomial::PolynomialDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_CONVERSION_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_
