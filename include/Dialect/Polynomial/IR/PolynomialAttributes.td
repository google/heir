#ifndef INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
#define INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_

include "include/Dialect/Polynomial/IR/PolynomialDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

class Polynomial_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Polynomial_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_Attr : Polynomial_Attr<"Polynomial", "polynomial"> {
  let summary = "An attribute containing a single-variable polynomial.";
  let description = [{
     #poly = #_polynomial.polynomial<x**1024 + 1>
  }];

  let parameters = (ins "Polynomial":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Polynomial":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = Polynomial;
    Polynomial getPolynomial() const { return getValue(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}

def Ring_Attr : Polynomial_Attr<"Ring", "ring"> {
  let summary = "An attribute specifying a ring.";
  let description = [{
    An attribute specifying a polynomial quotient ring with integer
    coefficients, $\mathbb{Z}/n\mathbb{Z}[x] / (p(x))$.

    `cmod` is the coefficient modulus $n$, and `ideal` is the ring ideal
    $(p(x))$. Because all ideals in a single-variable polynomial ring are
    principal, the ideal is defined by a single polynomial.

      #ring = #_polynomial.ring<cmod=1234, ideal=#polynomial.polynomial<x**1024 + 1>>
  }];

  let parameters = (ins "IntegerAttr": $cmod, "Polynomial":$ideal);

  let builders = [
    AttrBuilderWithInferredContext<
        (ins "const APInt &": $cmod, "Polynomial":$ideal), [{
      return $_get(
        ideal.getContext(),
        IntegerAttr::get(IntegerType::get(ideal.getContext(), cmod.getBitWidth()), cmod),
        ideal
      );
    }]>
  ];
  let extraClassDeclaration = [{
    Polynomial ideal() const { return getIdeal(); }
    APInt coefficientModulus() const { return getCmod().getValue(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}


#endif  // INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
