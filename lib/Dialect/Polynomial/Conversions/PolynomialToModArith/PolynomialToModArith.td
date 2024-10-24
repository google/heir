#ifndef LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_
#define LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_

include "mlir/Pass/PassBase.td"
include "lib/Dialect/ModArith/IR/ModArithDialect.td"

def PolynomialToModArith : Pass<"polynomial-to-mod-arith", "ModuleOp"> {
  let summary = "Lower `polynomial` to `mod_arith`.";

  let description = [{
    This pass lowers the `polynomial` dialect to their corresponding tensor
    data type and modular arithmetic operations of `mod_arith`.

    This pass requires the coefficients of the polynomial to be in canonical
    representation (implemented to be `[0, p)`). `polynomial.from_tensor` and
    `polynomial.constant` will enforce this constraint by applying the
    `mod_arith.reduce` op.
  }];
  let dependentDialects = [
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::polynomial::PolynomialDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_POLYNOMIAL_CONVERSIONS_POLYNOMIALTOMODARITH_POLYNOMIALTOMODARITH_TD_
