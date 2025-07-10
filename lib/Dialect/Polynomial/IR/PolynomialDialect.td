#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALDIALECT_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALDIALECT_TD_

include "mlir/IR/OpBase.td"

def Polynomial_Dialect : Dialect {
  let name = "polynomial";
  let cppNamespace = "::mlir::heir::polynomial";
  let description = [{
    The Polynomial dialect defines single-variable polynomial types and
    operations.

    The simplest use of `polynomial` is to represent mathematical operations in
    a polynomial ring `R[x]`, where `R` is another MLIR type like `i32`.

    More generally, this dialect supports representing polynomial operations in a
    quotient ring `R[X]/(f(x))` for some statically fixed polynomial `f(x)`.
    Two polyomials `p(x), q(x)` are considered equal in this ring if they have the
    same remainder when dividing by `f(x)`. When a modulus is given, ring operations
    are performed with reductions modulo `f(x)` and relative to the coefficient ring
    `R`.

    Examples:

    ```mlir
    // A constant polynomial in a ring with i32 coefficients and no polynomial modulus
    #ring = #polynomial.ring<coefficientType=i32>
    %a = polynomial.constant <1 + x**2 - 3x**3> : polynomial.polynomial<#ring>

    // A constant polynomial in a ring with i32 coefficients, modulo (x^1024 + 1)
    #modulus = #polynomial.int_polynomial<1 + x**1024>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#modulus>
    %a = polynomial.constant <1 + x**2 - 3x**3> : polynomial.polynomial<#ring>

    // A constant polynomial in a ring with i32 coefficients, with a polynomial
    // modulus of (x^1024 + 1) and a coefficient modulus of 17.
    #modulus = #polynomial.int_polynomial<1 + x**1024>
    !coeff_ty = !mod_arith.int<17 : i32>
    #ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#modulus>
    %a = polynomial.constant <1 + x**2 - 3x**3> : polynomial.polynomial<#ring>
    ```
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALDIALECT_TD_
