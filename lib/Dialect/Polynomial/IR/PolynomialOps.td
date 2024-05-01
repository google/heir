#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_TD_

include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "lib/Dialect/Polynomial/IR/PolynomialTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Polynomial_Op<string mnemonic, list<Trait> traits = []> :
        Op<Polynomial_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::polynomial";
}

class Polynomial_BinOp<string mnemonic, list<Trait> traits = []> :
        Polynomial_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType, ElementwiseMappable])> {
  let arguments = (ins PolynomialLike:$lhs, PolynomialLike:$rhs);
  let results = (outs PolynomialLike:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($output))" ;
}

def Polynomial_AddOp : Polynomial_BinOp<"add", [Commutative]> {
  let summary = "Addition operation between polynomials.";
}

def Polynomial_SubOp : Polynomial_BinOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
  let hasCanonicalizer = 1;
}

def Polynomial_MulOp : Polynomial_BinOp<"mul", [Commutative]> {
  let summary = "Multiplication operation between polynomials.";
}

def Polynomial_MulScalarOp : Polynomial_Op<"mul_scalar", [
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

def Polynomial_LeadingTermOp: Polynomial_Op<"leading_term"> {
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

def Polynomial_MonomialOp: Polynomial_Op<"monomial"> {
  let summary = "Create a polynomial that consists of a single monomial.";
  let arguments = (ins AnyInteger:$coefficient, Index:$degree);
  let results = (outs Polynomial:$output);
}

def Polynomial_MonomialMulOp: Polynomial_Op<"monomial_mul", [AllTypesMatch<["input", "output"]>]> {
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

def Polynomial_FromTensorOp : Polynomial_Op<"from_tensor", [Pure]> {
  let summary = "Creates a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `polynomial.from_tensor` creates a polynomial value from a tensor of coefficients.
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

def Polynomial_ToTensorOp : Polynomial_Op<"to_tensor", [Pure]> {
  let summary = "Creates a tensor containing the coefficients of a polynomial.";
  let description = [{
    `polynomial.to_tensor` creates a tensor value containing the coefficients of the
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

def Polynomial_ConstantOp : Polynomial_Op<"constant", [Pure]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins Polynomial_PolynomialAttr:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($output))";
}

def Polynomial_NTTOp : Polynomial_Op<"ntt", [Pure]> {
  let summary = "Computes point-value tensor representation of a polynomial.";
  let description = [{
    `polynomial.ntt` computes the forward integer Number Theoretic Transform
    (NTT) on the input polynomial. It returns a tensor containing a point-value
    representation of the input polynomial. The output tensor has shape equal to
    the degree of the ring's ideal generation polynomial. The polynomial's
    RingAttr is embedded as the encoding attribute of the output tensor.

    Given an input polynomial $F(x)$ (over a ring with degree $n$) and a
    primitive $n$-th root of unity $\omega_n$, the output is the list of $n$
    evaluations

    $f_k = F(\omega_n^k) ; k \in [0, n)$
    The choice of primitive root is determined by subsequent lowerings.
  }];

  let arguments = (ins Polynomial:$input);

  let results = (outs RankedTensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Polynomial_INTTOp : Polynomial_Op<"intt", [Pure]> {
  let summary = "Computes the reverse integer Number Theoretic Transform (NTT).";
  let description = [{
    `polynomial.intt` computes the reverse integer Number Theoretic Transform
    (INTT) on the input tensor. This is the inverse operation of the
    `polynomial.ntt` operation.

    The input tensor is interpreted as a point-value representation of the
    output polynomial at powers of a primitive $n$-th root of unity (see
    `polynomial.ntt`). The ring of the polynomial is taken from the required
    encoding attribute of the tensor.
  }];

  let arguments = (ins RankedTensorOf<[AnyInteger]>:$input);

  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_TD_
