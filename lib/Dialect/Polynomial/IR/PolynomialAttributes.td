#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_

include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/EnumAttr.td"

class Polynomial_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Polynomial_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_IntPolynomialAttr : Polynomial_Attr<"IntPolynomial", "int_polynomial"> {
  let summary = "an attribute containing a single-variable polynomial with integer coefficients";
  let description = [{
    A polynomial attribute represents a single-variable polynomial with integer
    coefficients, which is used to define the modulus of a `RingAttr`, as well
    as to define constants and perform constant folding for `polynomial` ops.

    The polynomial must be expressed as a list of monomial terms, with addition
    or subtraction between them. The choice of variable name is arbitrary, but
    must be consistent across all the monomials used to define a single
    attribute. The order of monomial terms is arbitrary, each monomial degree
    must occur at most once.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 + 1>
    ```
  }];
  let parameters = (ins "::mlir::heir::polynomial::IntPolynomial":$polynomial);
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_FloatPolynomialAttr : Polynomial_Attr<"FloatPolynomial", "float_polynomial"> {
  let summary = "an attribute containing a single-variable polynomial with double precision floating point coefficients";
  let description = [{
    A polynomial attribute represents a single-variable polynomial with double
    precision floating point coefficients.

    The polynomial must be expressed as a list of monomial terms, with addition
    or subtraction between them. The choice of variable name is arbitrary, but
    must be consistent across all the monomials used to define a single
    attribute. The order of monomial terms is arbitrary, each monomial degree
    must occur at most once.

    Example:

    ```mlir
    #poly = #polynomial.float_polynomial<0.5 x**7 + 1.5>
    ```
  }];
  let parameters = (ins "FloatPolynomial":$polynomial);
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_TypedIntPolynomialAttr : Polynomial_Attr<
    "TypedIntPolynomial", "typed_int_polynomial"> {
  let summary = "a typed int_polynomial";
  let description = [{
    Example:

    ```mlir
    !poly_ty = !polynomial.polynomial<ring=<coefficientType=i32>>
    #poly = int<1 x**7 + 4> : !poly_ty
    #poly_verbose = #polynomial.typed_int_polynomial<1 x**7 + 4> : !poly_ty
    ```
  }];
  let parameters = (ins "::mlir::Type":$type, "::mlir::heir::polynomial::IntPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const IntPolynomial &":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntPolynomialAttr::get(type.getContext(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Attribute &":$value), [{
      return $_get(type.getContext(), type, ::llvm::cast<IntPolynomialAttr>(value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::heir::polynomial::IntPolynomialAttr;
  }];
}

def Polynomial_TypedFloatPolynomialAttr : Polynomial_Attr<
    "TypedFloatPolynomial", "typed_float_polynomial"> {
  let summary = "a typed float_polynomial";
  let description = [{
    Example:

    ```mlir
    !poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
    #poly = float<1.4 x**7 + 4.5> : !poly_ty
    #poly_verbose = #polynomial.typed_float_polynomial<1.4 x**7 + 4.5> : !poly_ty
    ```
  }];
  let parameters = (ins "::mlir::Type":$type, "::mlir::heir::polynomial::FloatPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const FloatPolynomial &":$value), [{
      return $_get(
        type.getContext(),
        type,
        FloatPolynomialAttr::get(type.getContext(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Attribute &":$value), [{
      return $_get(type.getContext(), type, ::llvm::cast<FloatPolynomialAttr>(value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::heir::polynomial::FloatPolynomialAttr;
  }];
}

def Polynomial_ChebyshevPolynomialAttr : Polynomial_Attr<"ChebyshevPolynomial", "chebyshev_polynomial"> {
  let summary = "an attribute containing a single-variable polynomial with float coefficients in the Chebyshev basis";
  let description = [{
    This attribute represents a single-variable polynomial with double
    precision floating point coefficients, represented in the basis of
    Chebyshev polynomials of the first kind.

    Example:

    ```mlir
    #poly = #polynomial.chebyshev_polynomial<[1.0, 2.0, 3.0, 4.0]>
    ```
  }];
  let parameters = (ins "ArrayAttr":$coefficients);
  let assemblyFormat = "`<` $coefficients `>`";
}

def Polynomial_TypedChebyshevPolynomialAttr : Polynomial_Attr<
    "TypedChebyshevPolynomial", "typed_chebyshev_polynomial"> {
  let summary = "a typed chebyshev_polynomial";
  let description = [{
    Example:

    ```mlir
    !poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
    #poly = float<1.4 x**7 + 4.5> : !poly_ty
    #poly_verbose = #polynomial.typed_chebyshev_polynomial<[1.4, 4.5, 6.0, 7.0]> : !poly_ty
    ```
  }];
  let parameters = (ins "::mlir::Type":$type, "::mlir::heir::polynomial::ChebyshevPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const ArrayAttr &":$value), [{
      return $_get(
        type.getContext(),
        type,
        ChebyshevPolynomialAttr::get(type.getContext(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const ChebyshevPolynomial &":$value), [{
      return $_get(
        type.getContext(),
        type,
        ChebyshevPolynomialAttr::get(type.getContext(), value.getCoefficientsArrayAttr(type.getContext())));
    }]>,
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::heir::polynomial::ChebyshevPolynomialAttr;
  }];
}

def Polynomial_RingAttr : Polynomial_Attr<"Ring", "ring", [OpAsmAttrInterface]> {
  let summary = "an attribute specifying a polynomial ring";
  let description = [{
    A ring describes the domain in which polynomial arithmetic occurs. The ring
    attribute in `polynomial` represents the more specific case of polynomials
    with a single indeterminate; whose coefficients can be represented by
    another MLIR type (`coefficientType`).

    All semantics pertaining to arithmetic in the ring must be owned by the
    coefficient type. For example, if the polynomials are with integer
    coefficients taken modulo a prime $p$, then `coefficientType` must be a
    type that represents integers modulo $p$, such as `mod_arith<p>`.

    Additionally, a polynomial ring may specify a `polynomialModulus`, which
    converts polynomial arithmetic to the analogue of modular integer
    arithmetic, where each polynomial is represented as its remainder when
    dividing by the modulus. For single-variable polynomials, a polynomial
    modulus is always specified via a single polynomial.

    An expressive example is polynomials with i32 coefficients, whose
    coefficients are taken modulo `2**32 - 5`, with a polynomial modulus of
    `x**1024 - 1`.

    ```mlir
    #poly_mod = #polynomial.int_polynomial<-1 + x**1024>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly_mod>

    %0 = ... : polynomial.polynomial<#ring>
    ```

    In this case, the value of a polynomial is always "converted" to a
    canonical form by applying repeated reductions by setting `x**1024 = 1`
    and simplifying.
  }];

  let parameters = (ins
    "Type": $coefficientType,
    OptionalParameter<"::mlir::heir::polynomial::IntPolynomialAttr">: $polynomialModulus
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilderWithInferredContext<
        (ins "::mlir::Type":$coefficientTy,
              CArg<"::mlir::heir::polynomial::IntPolynomialAttr", "nullptr"> :$polynomialModulusAttr), [{
      return $_get(
        coefficientTy.getContext(),
        coefficientTy,
        polynomialModulusAttr);
    }]>,
  ];

  let extraClassDeclaration = [{
    // Helper method for other Type/Attribute containing this Attribute.
    void getAliasSuffix(::llvm::raw_ostream& os) const;

    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const;
  }];
}

def Polynomial_PrimitiveRootAttr: Polynomial_Attr<"PrimitiveRoot", "primitive_root"> {
  let summary = "an attribute containing a typed root value and its degree as a root of unity";
  let description = [{
    A primitive root attribute stores a typed root `value` and an integer
    `degree`, corresponding to a primitive root of unity of the given degree.

    The root value is represented as either:
    - `#mod_arith.value<...>` for a single modular coefficient ring, or
    - `#rns.value<...>` for an RNS coefficient ring.

    This is used as an attribute on `polynomial.ntt` and `polynomial.intt` ops
    to specify the root of unity used in lowering the transform.

    Example:

    ```mlir
    #v = #mod_arith.value<type=!mod_arith.int<17 : i32>, value=3 : i32>
    #poly = #polynomial.primitive_root<value=#v, degree=8 : i32>
    ```
  }];
  let parameters = (ins
    "::mlir::Attribute":$value,
    "IntegerAttr":$degree
  );
  let assemblyFormat = "`<` struct(params) `>`";
}


#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
