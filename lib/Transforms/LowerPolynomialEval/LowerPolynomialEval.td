#ifndef LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_LOWERPOLYNOMIALEVAL_TD_
#define LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_LOWERPOLYNOMIALEVAL_TD_

include "mlir/Pass/PassBase.td"

def LowerPolynomialEval : Pass<"lower-polynomial-eval"> {
  let summary = "Lowers the polynomial.eval operation";
  let description = [{
  This pass lowers the `polynomial.eval` operation to a sequence of arithmetic
  operations in the relevant dialect.

  Dialects that wish to support this pass must implement the
  `DialectPolynomialEvalInterface` dialect interface, which informs this pass
  what operations in the target dialect correspond to scalar multiplication and
  addition, as well as how to properly materialize constants as values.

  This pass supports multiple options for lowering a `polynomial.eval` op,
  including the following. The required basis representation of the polynomial
  is listed alongside each method. The chosen method is controlled by the
  `method` pass option, which defaults to automatically select the method.

    - `"horner"`: Horner's method (monomial basis)
    - `"ps"` Paterson-Stockmeyer (monomial basis)
    - `"pscheb"`: Paterson-Stockmeyer (Chebyshev basis)

  // TODO(#1565): Add support for Chebyshev-basis methods
  //  - `"clenshaw"`: Clenshaw's method (Chebyshev basis)
  //  - `"bsgs"`: Baby Step Giant Step (Chebyshev basis)
  }];
  let dependentDialects = [
    "::mlir::heir::polynomial::PolynomialDialect",
  ];
  let options = [
    Option<"method", "method", "mlir::heir::PolynomialApproximationMethod",
          /*default=*/"mlir::heir::PolynomialApproximationMethod::Automatic",
          "The method used to lower polynomial.eval",
          [{::llvm::cl::values(
                clEnumValN(mlir::heir::PolynomialApproximationMethod::Automatic,
                           "auto", "Automatically selected"),
                clEnumValN(mlir::heir::PolynomialApproximationMethod::Horner,
                           "horner", "Horner's method (monomial basis)"),
                clEnumValN(mlir::heir::PolynomialApproximationMethod::PatersonStockmeyer,
                           "ps", "Paterson-Stockmeyer method (monomial basis)"),
                clEnumValN(mlir::heir::PolynomialApproximationMethod::PatersonStockmeyerChebyshev,
                           "pscheb", "Paterson-Stockmeyer method (Chebyshev basis)")
          )}]>,
    Option<"minCoefficientThreshold", "min-coefficient-threshold", "double",
          /*default=*/"1e-12",
          "Minimum threshold for coefficients to be included in the lowered polynomial. "
          "Coefficients with absolute value below this threshold will be dropped.">,
  ];
}

#endif  // LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_LOWERPOLYNOMIALEVAL_TD_
