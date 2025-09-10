#ifndef LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_
#define LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_

include "mlir/Pass/PassBase.td"

def PolynomialToModArith : Pass<"polynomial-to-mod-arith", "ModuleOp"> {
  let summary = "Lower `polynomial` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `polynomial` dialect to standard MLIR plus mod_arith,
    including possibly ops from affine, tensor, linalg, and arith.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];

  // TODO(#2157): Remove option once key_switch_inner lowering is done
  let options = [
    Option<"buildMaterializations", "build-materializations", "bool", /*default=*/"true",
           "Whether to build materializations">,
  ];
}

#endif  // LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_
