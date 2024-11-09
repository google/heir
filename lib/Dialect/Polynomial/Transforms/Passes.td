#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PolyMulToNTT : Pass<"convert-polynomial-mul-to-ntt"> {
  let summary = "Rewrites polynomial operations to their NTT equivalents";
  let description = [{
    Applies a rewrite pattern to convert polynomial multiplication to the
    equivalent using the number-theoretic transforms (NTT) when possible.

    Polynomial multiplication can be rewritten as polynomial.NTT
    on each operand, followed by modulo elementwise multiplication of the
    point-value representation and then the inverse-NTT back to coefficient
    representation.
  }];
  let dependentDialects = ["mlir::polynomial::PolynomialDialect", "heir::mod_arith::ModArithDialect"];
}

def PolyPopulatePrimitiveRoot : Pass<"polynomial-populate-primitive-root"> {
  let summary = "Populate primitive root for polynomial NTT/iNTT operations";
  let description = [{
    Choose a primitive root for NTT/iNTT operations whose root attribute was
    not specified.

    Currently the existence and choice of the primitive root of a prime is
    looked up from a table maintained in StaticRoots.h.
  }];
  let dependentDialects = ["mlir::polynomial::PolynomialDialect"];
}


#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
