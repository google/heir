#ifndef INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
#define INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_

include "include/Dialect/Polynomial/IR/PolynomialDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

class Polynomial_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Polynomial_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_Attr : Polynomial_Attr<"Polynomial", "polynomial"> {
  let summary = "An attribute containing a single-variable polynomial.";
  let description = [{
     #poly = #polynomial.poly<x**1024 + 1>
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

      #ring = #polynomial.ring<cmod=1234, ideal=#polynomial.polynomial<x**1024 + 1>>
  }];

  // The extra cwidth parameter is required because, when the MLIR framework's
  // StorageUniquer is determining whether to build a new instance of this
  // attribute or reuse an existing one, it does an == comparison on the
  // parameters, and comparing two APInts with different underlying bit widths
  // leads to an assertion failure. By adding the container bit width as a
  // parameter, that bit width is compared first and the rest of the check is
  // skipped if the bit widths disagree. The custom builder and custom
  // parser/printer ensures that cwidth is automatically inferred from cmod, and
  // not printed or parsed, so that the cwidth parameter need not be known to the
  // user.
  let parameters = (ins "int64_t": $cwidth, "APInt": $cmod, "Polynomial":$ideal);

  let builders = [
    AttrBuilderWithInferredContext<(ins "APInt": $cmod, "Polynomial":$ideal), [{
      return $_get(ideal.getContext(), cmod.getBitWidth(), cmod, ideal);
    }]>
  ];
  let extraClassDeclaration = [{
    Polynomial ideal() const { return getIdeal(); }
    APInt coefficientModulus() const { return getCmod(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}


#endif  // INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
