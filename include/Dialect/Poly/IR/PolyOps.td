#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_

include "include/Dialect/Poly/IR/PolyAttributes.td"
include "include/Dialect/Poly/IR/PolyDialect.td"
include "include/Dialect/Poly/IR/PolyTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Poly_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poly_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::poly";
}

class Poly_BinOp<string mnemonic, list<Trait> traits = [Pure, SameOperandsAndResultType, Commutative, ElementwiseMappable]> :
        Poly_Op<mnemonic, traits> {
  let arguments = (ins PolynomialLike:$lhs, PolynomialLike:$rhs);
  let results = (outs PolynomialLike:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($output))" ;
}

def Poly_AddOp : Poly_BinOp<"add"> {
  let summary = "Addition operation between polynomials.";
}

def Poly_SubOp : Poly_BinOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
  let hasCanonicalizer = 1;
}

def Poly_MulOp : Poly_BinOp<"mul"> {
  let summary = "Multiplication operation between polynomials.";
}

def Poly_MulScalarOp : Poly_Op<"mul_scalar", [
      ElementwiseMappable, AllTypesMatch<["polynomial", "output"]>]> {
  let summary = "Multiplication by a scalar of the field.";

  let arguments = (ins
    PolynomialLike:$polynomial,
    AnyInteger:$scalar
  );

  let results = (outs
    PolynomialLike:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($polynomial)) `,` type($scalar)";
}

def Poly_LeadingTermOp: Poly_Op<"leading_term"> {
  let summary = "Compute the leading term of the polynomial.";
  let description = [{
    The degree of a polynomial is the largest $k$ for which the coefficient
    $a_k$ of $x^k$ is nonzero. The leading term is the term $a_k x^k$, which
    this op represents as a pair of results.
  }];
  let arguments = (ins Polynomial:$input);
  let results = (outs Index:$degree, AnyInteger:$coefficient);
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` `(` type($degree) `,` type($coefficient) `)`";
}

def Poly_MonomialOp: Poly_Op<"monomial"> {
  let summary = "Create a polynomial that consists of a single monomial.";
  let arguments = (ins AnyInteger:$coefficient, Index:$degree);
  let results = (outs Polynomial:$output);
}

def Poly_MonomialMulOp: Poly_Op<"monomial_mul", [AllTypesMatch<["input", "output"]>]> {
  let summary = "Multiply a polynomial by a monic monomial.";
  let description = [{
    In the ring of polynomials mod $x^n - 1$, `monomial_mul` can be interpreted
    as a cyclic shift of the coefficients of the polynomial. For some rings,
    this results in optimized lowerings that involve rotations and rescaling
    of the coefficients of the input.
  }];
  let arguments = (ins Polynomial:$input, Index:$monomialDegree);
  let results = (outs Polynomial:$output);
  let hasVerifier = 1;
}

def Poly_FromTensorOp : Poly_Op<"from_tensor", [Pure]> {
  let summary = "Creates a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `poly.from_tensor` creates a polynomial value from a tensor of coefficients.
    The input tensor must list the coefficients in degree-increasing order.

    The input one-dimensional tensor may have size at most the degree of the
    ring's ideal generator polynomial, with smaller dimension implying that
    all higher-degree terms have coefficient zero.
  }];
  let arguments = (ins RankedTensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";

  let builders = [
    // Builder that infers coefficient modulus from tensor bit width,
    // and uses whatever input ring is provided by the caller.
    OpBuilder<(ins "::mlir::Value":$input, "RingAttr":$ring)>
  ];
  let hasVerifier = 1;
}

def Poly_ToTensorOp : Poly_Op<"to_tensor", [Pure]> {
  let summary = "Creates a tensor containing the coefficients of a polynomial.";
  let description = [{
    `poly.to_tensor` creates a tensor value containing the coefficients of the
    input polynomial. The output tensor contains the coefficients in
    degree-increasing order.

    Operations that act on the coefficients of a polynomial, such as extracting
    a specific coefficient or extracting a range of coefficients, should be
    implemented by composing `to_tensor` with the relevant `tensor` dialect
    ops.

    The output tensor has shape equal to the degree of the ring's ideal
    generator polynomial, including zeroes.
  }];
  let arguments = (ins Polynomial:$input);
  let results = (outs RankedTensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";

  let hasVerifier = 1;
}

def Poly_ConstantOp : Poly_Op<"constant", [Pure]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins Polynomial_Attr:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($output))";
}

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
