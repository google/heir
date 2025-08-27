#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_TD_

include "lib/Dialect/ModArith/IR/ModArithTypes.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "lib/Dialect/Polynomial/IR/PolynomialTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Polynomial_Op<string mnemonic, list<Trait> traits = []> :
    Op<Polynomial_Dialect, mnemonic, traits # [Pure]> {
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class Polynomial_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Polynomial_PolynomialType:$operand);
  let results = (outs Polynomial_PolynomialType:$result);
}

class Polynomial_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType, ElementwiseMappable])> {
  let arguments = (ins PolynomialLike:$lhs, PolynomialLike:$rhs);
  let results = (outs PolynomialLike:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def Polynomial_AddOp : Polynomial_BinaryOp<"add", [Commutative]> {
  let summary = "Addition operation between polynomials.";
  let description = [{
    Performs polynomial addition on the operands. The operands may be single
    polynomials or containers of identically-typed polynomials, i.e., polynomials
    from the same underlying ring with the same coefficient types.

    This op is defined to occur in the ring defined by the ring attribute of
    the two operands, meaning the arithmetic is taken modulo the
    polynomialModulus of the ring as well as modulo any semantics defined by
    the coefficient type.

    Example:

    ```mlir
    // add two polynomials modulo x^1024 - 1
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
    %1 = polynomial.constant int<x**5 - x + 1> : !polynomial.polynomial<#ring>
    %2 = polynomial.add %0, %1 : !polynomial.polynomial<#ring>
    ```
  }];
}

def Polynomial_SubOp : Polynomial_BinaryOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
  let description = [{
    Performs polynomial subtraction on the operands. The operands may be single
    polynomials or containers of identically-typed polynomials, i.e., polynomials
    from the same underlying ring with the same coefficient types.

    This op is defined to occur in the ring defined by the ring attribute of
    the two operands, meaning the arithmetic is taken modulo the
    polynomialModulus of the ring as well as modulo any semantics defined by
    the coefficient type.

    Example:

    ```mlir
    // subtract two polynomials modulo x^1024 - 1
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
    %1 = polynomial.constant int<x**5 - x + 1> : !polynomial.polynomial<#ring>
    %2 = polynomial.sub %0, %1 : !polynomial.polynomial<#ring>
    ```
  }];
}

def Polynomial_MulOp : Polynomial_BinaryOp<"mul", [Commutative]> {
  let summary = "Multiplication operation between polynomials.";
  let description = [{
    Performs polynomial multiplication on the operands. The operands may be single
    polynomials or containers of identically-typed polynomials, i.e., polynomials
    from the same underlying ring with the same coefficient types.

    This op is defined to occur in the ring defined by the ring attribute of
    the two operands, meaning the arithmetic is taken modulo the
    polynomialModulus of the ring as well as modulo any semantics defined by
    the coefficient type.

    Example:

    ```mlir
    // multiply two polynomials modulo x^1024 - 1
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
    %1 = polynomial.constant int<x**5 - x + 1> : !polynomial.polynomial<#ring>
    %2 = polynomial.mul %0, %1 : !polynomial.polynomial<#ring>
    ```
  }];
}

def Polynomial_MulScalarOp : Polynomial_Op<"mul_scalar", [
      ElementwiseMappable, AllTypesMatch<["polynomial", "output"]>]> {
  let summary = "Multiplication by a scalar of the field.";
  let description = [{
    Multiplies the polynomial operand's coefficients by a given scalar value.
    The `scalar` input must have the same type as the polynomial ring's
    coefficientType.

    Example:

    ```mlir
    // multiply two polynomials modulo x^1024 - 1
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
    %1 = arith.constant 3 : i32
    %2 = polynomial.mul_scalar %0, %1 : !polynomial.polynomial<#ring>, i32
    ```
  }];

  let arguments = (ins
    PolynomialLike:$polynomial,
    AnyType:$scalar
  );
  let results = (outs
    PolynomialLike:$output
  );
  let assemblyFormat = "operands attr-dict `:` type($polynomial) `,` type($scalar)";
  let hasVerifier = 1;
}

def Polynomial_LeadingTermOp: Polynomial_Op<"leading_term"> {
  let summary = "Compute the leading term of the polynomial.";
  let description = [{
    The degree of a polynomial is the largest $k$ for which the coefficient
    `a_k` of `x^k` is nonzero. The leading term is the term `a_k * x^k`, which
    this op represents as a pair of results. The first is the degree `k` as an
    index, and the second is the coefficient, whose type matches the
    coefficient type of the polynomial's ring attribute.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
    %1, %2 = polynomial.leading_term %0 : !polynomial.polynomial<#ring> -> (index, i32)
    ```
  }];
  let arguments = (ins Polynomial_PolynomialType:$input);
  let results = (outs Index:$degree, AnyType:$coefficient);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` `(` type($degree) `,` type($coefficient) `)`";
  let hasVerifier = 1;
}

def Polynomial_MonomialOp: Polynomial_Op<"monomial"> {
  let summary = "Create a polynomial that consists of a single monomial.";
  let description = [{
    Construct a polynomial that consists of a single monomial term, from its
    degree and coefficient as dynamic inputs.

    The coefficient type of the output polynomial's ring attribute must match
    the `coefficient` input type.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %deg = arith.constant 1023 : index
    %five = arith.constant 5 : i32
    %0 = polynomial.monomial %five, %deg : (i32, index) -> !polynomial.polynomial<#ring>
    ```
  }];
  let arguments = (ins AnyType:$coefficient, Index:$degree);
  let results = (outs Polynomial_PolynomialType:$output);
  let hasVerifier = 1;
}

def Polynomial_MonicMonomialMulOp: Polynomial_Op<"monic_monomial_mul", [AllTypesMatch<["input", "output"]>]> {
  let summary = "Multiply a polynomial by a monic monomial.";
  let description = [{
    Multiply a polynomial by a monic monomial, meaning a polynomial of the form
    `1 * x^k` for an index operand `k`.

    In some special rings of polynomials, such as a ring of polynomials
    modulo `x^n - 1`, `monomial_mul` can be interpreted as a cyclic shift of
    the coefficients of the polynomial. For some rings, this results in
    optimized lowerings that involve rotations and rescaling of the
    coefficients of the input.
  }];
  let arguments = (ins PolynomialLike:$input, Index:$monomialDegree);
  let results = (outs PolynomialLike:$output);
}

def Polynomial_FromTensorOp : Polynomial_Op<"from_tensor", [Pure]> {
  let summary = "Creates a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `polynomial.from_tensor` creates a polynomial value from a tensor of coefficients.
    The input tensor must list the coefficients in degree-increasing order.

    The input one-dimensional tensor may have size at most the degree of the
    ring's polynomialModulus generator polynomial, with smaller dimension implying that
    all higher-degree terms have coefficient zero.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %two = arith.constant 2 : i32
    %five = arith.constant 5 : i32
    %coeffs = tensor.from_elements %two, %two, %five : tensor<3xi32>
    %poly = polynomial.from_tensor %coeffs : tensor<3xi32> -> !polynomial.polynomial<#ring>
    ```
  }];
  let arguments = (ins RankedTensorOf<[AnyType]>:$input);
  let results = (outs Polynomial_PolynomialType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";

  let builders = [
    // Builder that infers coefficient modulus from tensor bit width,
    // and uses whatever input ring is provided by the caller.
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::heir::polynomial::RingAttr":$ring)>
  ];
  let hasVerifier = 1;
}

def Polynomial_ToTensorOp : Polynomial_Op<"to_tensor", [Pure]> {
  let summary = "Creates a tensor containing the coefficients of a polynomial.";
  let description = [{
    `polynomial.to_tensor` creates a dense tensor value containing the
    coefficients of the input polynomial. The output tensor contains the
    coefficients in degree-increasing order.

    Operations that act on the coefficients of a polynomial, such as extracting
    a specific coefficient or extracting a range of coefficients, should be
    implemented by composing `to_tensor` with the relevant `tensor` dialect
    ops.

    The output tensor has shape equal to the degree of the polynomial ring
    attribute's polynomialModulus, including zeroes.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    %two = arith.constant 2 : i32
    %five = arith.constant 5 : i32
    %coeffs = tensor.from_elements %two, %two, %five : tensor<3xi32>
    %poly = polynomial.from_tensor %coeffs : tensor<3xi32> -> !polynomial.polynomial<#ring>
    %tensor = polynomial.to_tensor %poly : !polynomial.polynomial<#ring> -> tensor<1024xi32>
    ```
  }];
  let arguments = (ins Polynomial_PolynomialType:$input);
  let results = (outs RankedTensorOf<[AnyType]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def Polynomial_ModSwitchOp : Polynomial_Op<"mod_switch", [Pure]> {
  let summary = "Change the coefficient type of a polynomial.";
  let description = [{
    `polynomial.mod_switch` changes the coefficient type of a polynomial.

    The two polynomials must have the same polynomialModulus.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 - 1>
    #ring32 = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
    #ring64 = #polynomial.ring<coefficientType=i64, polynomialModulus=#poly>
    %poly = polynomial.mod_switch %coeffs : !polynomial.polynomial<#ring64> to !polynomial.polynomial<#ring32>
    ```
  }];
  let arguments = (ins Polynomial_PolynomialType:$input);
  let results = (outs Polynomial_PolynomialType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}


def Polynomial_AnyTypedPolynomialAttr : AnyAttrOf<[
  Polynomial_TypedFloatPolynomialAttr,
  Polynomial_TypedIntPolynomialAttr,
  Polynomial_TypedChebyshevPolynomialAttr
]>;

def Polynomial_ConstantOp : Op<Polynomial_Dialect, "constant",
    [Pure, InferTypeOpAdaptor]> {
  let summary = "Define a constant polynomial via an attribute.";
  let description = [{
    Example:

    ```mlir
    !int_poly_ty = !polynomial.polynomial<ring=<coefficientType=i32>>
    %0 = polynomial.constant int<1 + x**2> : !int_poly_ty

    !float_poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
    %1 = polynomial.constant float<0.5 + 1.3e06 x**2> : !float_poly_ty
    ```
  }];
  let arguments = (ins Polynomial_AnyTypedPolynomialAttr:$value);
  let results = (outs Polynomial_PolynomialType:$output);
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_NTTOp : Polynomial_Op<"ntt", [Pure]> {
  let summary = "Computes point-value tensor representation of a polynomial.";
  let description = [{
    `polynomial.ntt` computes the forward integer Number Theoretic Transform
    (NTT) on the input polynomial. It returns a tensor containing a point-value
    representation of the input polynomial. The output tensor has shape equal
    to the degree of the ring's `polynomialModulus`. The polynomial's RingAttr
    is embedded as the encoding attribute of the output tensor.

    Given an input polynomial `F(x)` over a ring whose `polynomialModulus` has
    degree `n`, and a primitive `n`-th root of unity `omega_n`, the output is
    the list of $n$ evaluations

      `f[k] = F(omega[n]^k) ; k = {0, ..., n-1}`

    The choice of primitive root may be optionally specified.
  }];
  let arguments = (ins
    Polynomial_PolynomialType:$input,
    OptionalAttr<Polynomial_PrimitiveRootAttr>:$root
  );
  let results = (outs RankedTensorOf<[ModArith_ModArithType]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Polynomial_INTTOp : Polynomial_Op<"intt", [Pure]> {
  let summary = "Computes the reverse integer Number Theoretic Transform (NTT).";
  let description = [{
    `polynomial.intt` computes the reverse integer Number Theoretic Transform
    (INTT) on the input tensor. This is the inverse operation of the
    `polynomial.ntt` operation.

    The input tensor is interpreted as a point-value representation of the
    output polynomial at powers of a primitive `n`-th root of unity (see
    `polynomial.ntt`). The ring of the polynomial is taken from the required
    encoding attribute of the tensor.

    The choice of primitive root may be optionally specified.
  }];
  let arguments = (
    ins RankedTensorOf<[ModArith_ModArithType]>:$input,
    OptionalAttr<Polynomial_PrimitiveRootAttr>:$root
  );
  let results = (outs Polynomial_PolynomialType:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Polynomial_EvalOp : Polynomial_Op<"eval", [AllTypesMatch<["value", "output"]>, ElementwiseMappable]> {
  let summary = "Evaluate a static polynomial attribute at a given SSA value.";
  let description = [{
      Evaluates the result of a polynomial specified as a static attribute at
      a given SSA value. The result represents the evaluation of the
      polynomial at the input value and produces a corresponding scalar
      value.

      Example:

      ```mlir
      !poly_ty = !polynomial.polynomial<ring=<coefficientType=i32>>
      #poly = #polynomial.typed_int_polynomial<1 + x + x**2> : !poly_ty
      %x = arith.constant 5 : i32
      %result = polynomial.eval #poly, %x : i32
      ```

      The coefficient type of the polynomial does not necessarily need to be
      the same as the scalar input type. For example, one may evaluate a
      square matrix in a polynomial, because the scalar-matrix operation is
      well-defined. It is the responsibility of the lowering to determine
      if the input is compatible with the polynomial coefficient type.
  }];
  let arguments = (ins
      Polynomial_AnyTypedPolynomialAttr: $polynomial,
      AnyType: $value
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "$polynomial `,` $value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Polynomial_KeySwitchInnerOp : Polynomial_Op<"key_switch_inner",
    [AllTypesMatch<["value", "constantOutput", "linearOutput"]>]> {
  let summary = "Key switch on an RNS polynomial component";
  let description = [{
    Generates a linear ciphertext that, when added to a ciphertext encrypted
    under the input key, switches it to a ciphertext encrypted under the output
    key.

    This operation is intended to be an internal implementation detail of
    higher-level ciphertext operations such as `ckks.relinearize`, isolated
    here for reuse among multiple op lowerings.
  }];
  let arguments = (ins
      Polynomial_PolynomialType:$value,
      RankedTensorOf<[Polynomial_PolynomialType]>: $keySwitchingKey
  );
  let results = (outs
    Polynomial_PolynomialType:$constantOutput,
    Polynomial_PolynomialType:$linearOutput
  );
  let hasVerifier = 1;
}



#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_TD_
