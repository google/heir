#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYATTRIBUTES_TD_

include "PolyDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

class Poly_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Poly_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_Attr : Poly_Attr<"Polynomial", "polynomial"> {
  let summary = "An attribute containing a single-variable polynomial.";
  let description = [{
     #poly = #poly.poly<x**1024 + 1>
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

def Ring_Attr : Poly_Attr<"Ring", "ring"> {
  let summary = "An attribute specifying a ring.";
  let description = [{
    An attribute specifying a polynomial quotient ring with integer
    coefficients, $\mathbb{Z}/n\mathbb{Z}[x] / (p(x))$.

    `cmod` is the coefficient modulus $n$, and `ideal` is the ring ideal
    $(p(x))$. Because all ideals in a single-variable polynomial ring are
    principal, the ideal is defined by a single polynomial.

      #ring = #poly.ring<cmod=1234, ideal=#poly.polynomial<x**1024 + 1>>
  }];

  let parameters = (ins "APInt": $cmod, "Polynomial":$ideal);

  let builders = [
    AttrBuilderWithInferredContext<(ins "APInt": $cmod, "Polynomial":$ideal), [{
      return $_get(ideal.getContext(), cmod, ideal);
    }]>
  ];
  let extraClassDeclaration = [{
    Polynomial ideal() const { return getIdeal(); }
    APInt coefficientModulus() const { return getCmod(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}


#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
