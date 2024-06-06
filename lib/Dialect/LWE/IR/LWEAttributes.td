#ifndef HEIR_LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
#define HEIR_LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class LWE_EncodingAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LWE_Dialect, attrName, traits # [
    // All encoding attributes are required to be compatible with a tensor
    // with an element type relevant to that encoding.
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>
]> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

class LWE_EncodingAttrWithScalingFactor<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LWE_EncodingAttr<attrName, attrMnemonic, traits> {
  // These parameters represent the base-2 logarithm of the scaling factor to
  // scale cleartexts in preparation for noise growth of FHE schemes. This
  // representation restricts the scaling factors to being powers of two.
  let parameters = (ins
    "unsigned":$cleartext_start,
    "unsigned":$cleartext_bitwidth
  );
}

def LWE_BitFieldEncoding
  : LWE_EncodingAttrWithScalingFactor<"BitFieldEncoding", "bit_field_encoding"> {
  let summary = "An attribute describing encoded LWE plaintexts using bit fields.";
  let description = [{
    A bit field encoding of an integer describes which contiguous region
    of bits a small integer occupies within a larger integer.

    The data describing the encoding consists of the starting bit positions of
    the cleartext bit field and its width, where the LSB is bit 0 and the MSB
    is bit `bit_width-1`. So the above example would have starting bit `30` and
    width `3`. The bits not specified for the message have semantics defined
    by the scheme or lowering.

    Note that this encoding does not specify the underlying bit width of the
    plaintext space. This is left for lowerings to decide.

    The presence of this attribute as the `encoding` attribute of a tensor
    indicates that the tensor is an LWE ciphertext.

    **Example (CGGI):**

    ```
    #encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
    !plaintext = !lwe.lwe_plaintext<encoding = #encoding>

    %0 = arith.constant 4 : i3
    %1 = lwe.encode %0 { encoding = #encoding }: i3 to !plaintext
    ```

    The above represents an LWE plaintext encoding the 3-bit cleartext 4 as an
    LWE ciphertext in a 32-bit integer, with a single bit of padding at the MSB.
    This corresponds to the following, where 0 denotes a 0 bit, `b` denotes a
    bit of the cleartext, `n` denotes a bit reserved for noise, and `|` is a
    visual aid to show where the bit fields begin and end.

    ```
       0|bbb|nn...n
    MSB^          ^LSB
    ```

    **Example (BGV):**

    Note: BGV uses the RLWE encodings, but they have the same bit-field encoding
    attributes as here. So this example serves mainly to show how this attribute
    can be used to specify storing bits in the LSB of a plaintext.

    ```
    #encoding = #lwe.bit_field_encoding<cleartext_start=4, cleartext_bitwidth=4>
    !plaintext = !lwe.lwe_plaintext<encoding = #encoding>

    %0 = arith.constant 9 : i4
    %1 = lwe.encode %0 { encoding = #encoding }: i4 to !plaintext
    ```

    The above represents an LWE plaintext encoding a 4-bit cleartext as an
    LWE ciphertext in the least-significant bits of a larger integer.
    This corresponds to the following.

    ```
       nn...n|bbbb
    MSB^         ^LSB
    ```
  }];
}

def LWE_UnspecifiedBitFieldEncoding
  : LWE_EncodingAttr<"UnspecifiedBitFieldEncoding",
      "unspecified_bit_field_encoding"> {
  let summary = "An attribute describing unspecified bit field encodings.";
  let description = [{
    See LWE_BitFieldEncoding for a description of bit field encodings.

    This attribute describes an unspecified bit field encoding; this is where
    the starting bit position of the cleartext bit field is unspecified, but its
    width is fixed. A noise growth analysis should be performed to determine the
    optimal amount of bits needed for noise and padding to specify the bit field
    encodings starting bit position.

    Example:

    ```
    #lwe_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth=3>
    %lwe_ciphertext = arith.constant <[1,2,3,4]> : tensor<4xi32, #lwe_encoding>
    ```
  }];

  // These parameters represent unspecified encodings and simply hold the
  // cleartext bit width in preparation for specifying the encoding scaling
  // factors after noise growth analysis.
  let parameters = (ins
    "unsigned":$cleartext_bitwidth
  );
}

def AnyLWEEncodingAttr : AnyAttrOf<[LWE_BitFieldEncoding, LWE_UnspecifiedBitFieldEncoding]>;

def RLWE_PolynomialCoefficientEncoding
  : LWE_EncodingAttrWithScalingFactor<"PolynomialCoefficientEncoding", "polynomial_coefficient_encoding"> {
  let summary = "An attribute describing encoded RLWE plaintexts via coefficients.";
  let description = [{
    A coefficient encoding of a list of integers asserts that the coefficients
    of the polynomials contain the cleartexts, with the same semantics as
    `bit_field_encoding` for per-coefficient encodings.

    The presence of this attribute as the `encoding` attribute of a tensor of
    `poly.poly` indicates that the tensor is an RLWE ciphertext for some RLWE
    scheme that supports the coefficient encoding.

    See `bit_field_encoding` for the definition of the `cleartext_start` and
    `cleartext_bitwidth` fields.

    Example:

    ```
    #generator = #poly.polynomial<1 + x**1024>
    #ring = #poly.ring<cmod=65536, ideal=#generator>
    #coeff_encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>

    %poly1 = poly.from_tensor %coeffs1 : tensor<10xi16> -> !poly.poly<#ring>
    %poly2 = poly.from_tensor %coeffs2 : tensor<10xi16> -> !poly.poly<#ring>
    %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring>, #coeff_encoding>
    ```
  }];
}

def RLWE_PolynomialEvaluationEncoding
  : LWE_EncodingAttrWithScalingFactor<"PolynomialEvaluationEncoding", "polynomial_evaluation_encoding"> {
  let summary = "An attribute describing encoded RLWE plaintexts via evaluations at fixed points.";
  let description = [{
    A "evaluation encoding" of a list of integers $(v_1, \dots, v_n)$ asserts
    that $f(x_1 ) = v_1, \dots, f(x_n) = v_n$ for some implicit, but fixed and
    distinct choice of inputs $x_i$. The encoded values are also scaled by a
    scale factor, having the same semantics as `bit_field_encoding`, but
    applied entry-wise (to either the coefficient or evaluation representation).

    This attribute can be used in multiple ways:

    - On a `poly.poly`, it asserts that the polynomial has been transformed
      from an evaluation tensor.
    - On a tensor of `poly.poly`, it asserts that the tensor is an RLWE
      ciphertext for some RLWE scheme that supports the evaluation encoding.

    A typical workflow for the BFV/BGV schemes using this encoding would be
    to apply a INTT operation to the input list of cleartexts to convert from
    evaluation form to coefficient form, then encrypt the resulting polynomial
    in coefficient form, then apply NTT back to the evaluation form for faster
    multiplication of ciphertexts.

    The points chosen are fixed to be the powers of a primitive root of unity
    of the coefficient ring of the plaintext space, which allows one to use
    NTT/INTT to tansform quickly between the coefficient and evaluation forms.

    Example:

    ```
    #generator = #poly.polynomial<1 + x**1024>
    // note that the cmod should be chosen so as to ensure a primitive root of
    // unity exists in the multiplicative group (Z / cmod Z)^*
    #ring = #poly.ring<cmod=65536, ideal=#generator>
    #lwe_encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

    %evals = arith.constant <[1, 2, 4, 5]> : tensor<4xi16>
    %poly1 = poly.intt %evals : tensor<4xi16, #ring> -> !poly.poly<#ring, #eval_encoding>
    %poly2 = poly.intt %evals : tensor<4xi16, #ring> -> !poly.poly<#ring, #eval_encoding>
    %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring, #eval_encoding>>
    ```

    See `bit_field_encoding` for the definition of the `cleartext_start` and
    `cleartext_bitwidth` fields.
  }];
}

// TODO(#183): does it make sense to use the bit-field parameters here? It may
// not work sensibly because the rounding happens after scaling-post-INTT, so
// rounding doesn't happen via bit shifting, and this scheme might need a scaling
// factor that is not a power of two.
def RLWE_InverseCanonicalEmbeddingEncoding
  : LWE_EncodingAttrWithScalingFactor<"InverseCanonicalEmbeddingEncoding", "inverse_canonical_embedding_encoding"> {
  let summary = "An attribute describing encoded RLWE plaintexts via the rounded inverse canonical embedding.";
  let description = [{
    Let $n$ be the degree of the polynomials in the plaintext space. An
    "inverse canonical embedding encoding" of a list of real or complex values
    $v_1, \dots, v_{n/2}$ is (almost) the inverse of the following decoding
    map.

    Define a map $\tau_N$ that maps a polynomial $p \in \mathbb{Z}[x] / (x^N + 1)
    \to \mathbb{C}^{N/2}$ by evaluating it at the following $N/2$ points,
    where $\omega = e^{2 \pi i / 2N}$ is the primitive $2N$th root of unity:

    \[
      \omega, \omega^3, \omega^5, \dots, \omega^{N-1}
    \]

    Then the complete decoding operation is $\textup{Decode}(p) =
    (1/\Delta)\tau_N(p)$, where $\Delta$ is a scaling parameter and $\tau_N$ is
    the truncated canonical embedding above. The encoding operation is the
    inverse of the decoding operation, with some caveats explained below.

    The map $\tau_N$ is derived from the so-called _canonical embedding_
    $\tau$, though in the standard canonical embedding, we evaluate at all odd
    powers of the root of unity, $\omega, \omega^3, \dots, \omega^{2N-1}$. For
    polynomials in the slightly larger space $\mathbb{R}[x] / (x^N + 1)$, the
    image of the canonical embedding is the subspace $H \subset \mathbb{C}^N$
    defined by tuples $(z_1, \dots, z_N)$ such that $\overline{z_i} =
    \overline{z_{N-i+1}}$. Note that this property holds because polynomial
    evaluation commutes with complex conjugates, and the second half of the
    roots of unity evaluate are complex conjugates of the first half. The
    converse, that any such tuple with complex conjugate symmetry has an
    inverse under $\tau$ with all real coefficients, makes $\tau$ is a
    bijection onto $H$. $\tau$ and its inverse are explicitly computable as
    discrete Fourier Transforms.

    Because of the symmetry in canonical embedding for real polynomials, inputs
    to this encoding can be represented as a list of $N/2$ complex points, with
    the extra symmetric structure left implicit. $\tau_N$ and its inverse can
    also be explicitly computed without need to expand the vectors to length
    $N$.

    The rounding step is required to invert the decoding because, while
    cleartexts must be (implicitly) in the subspace $H$, they need not be the
    output of $\tau_N$ for an _integer_ polynomial. The rounding step ensures
    we can use integer polynomial plaintexts for the FHE operations. There are
    multiple rounding mechanisms, and this attribute does not specify which is
    used, because in theory two ciphertexts that have used different roundings
    are still compatible, though they may have different noise growth patterns.

    The scaling parameter $\Delta$ is specified by the `cleartext_start` and
    `cleartext_bitwidth` parameters, which are applied coefficient-wise using
    the same semantics as the `bit_field_encoding`.

    This attribute can be used in multiple ways:

    - On a `poly.poly`, it asserts that the polynomial has been transformed
      from a coefficient list using the canonical embedding.
    - On a tensor of `poly.poly`, it asserts that the tensor is an RLWE
      ciphertext for some RLWE scheme that supports the approximate embedding
      encoding.

    A typical flow for the CKKS scheme using this encoding would be to apply an
    inverse FFT operation to invert the canonical embedding to be a polynomial
    with real coefficients, then encrypt scale the resulting polynomial's
    coefficients according to the scaling parameters, then round to get integer
    coefficients.

    Example:

    ```
    #generator = #poly.polynomial<1 + x**1024>
    #ring = #poly.ring<cmod=65536, ideal=#generator>
    #lwe_encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

    %evals = arith.constant <[1, 2, 4, 5]> : tensor<4xi16>
    %poly1 = poly.intt %evals : tensor<4xi16, #ring> -> !poly.poly<#ring, #eval_encoding>
    %poly2 = poly.intt %evals : tensor<4xi16, #ring> -> !poly.poly<#ring, #eval_encoding>
    %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring, #eval_encoding>>
    ```

    See `bit_field_encoding` for the definition of the `cleartext_start` and
    `cleartext_bitwidth` fields.
  }];
}

def AnyRLWEEncodingAttr : AnyAttrOf<[RLWE_PolynomialCoefficientEncoding, RLWE_PolynomialEvaluationEncoding, RLWE_InverseCanonicalEmbeddingEncoding]>;

def LWE_LWEParams : AttrDef<LWE_Dialect, "LWEParams"> {
  let mnemonic = "lwe_params";

  let parameters = (ins "IntegerAttr": $cmod, "unsigned":$dimension);

  let assemblyFormat = "`<` struct(params) `>`";
}

def LWE_RLWEParams : AttrDef<LWE_Dialect, "RLWEParams"> {
  let mnemonic = "rlwe_params";
  let description = [{
    An attribute describing classical RLWE parameters:

    - `dimension`: the number of polynomials used in an RLWE sample, analogous
      to LWEParams.dimension.
    - `ring`: the polynomial ring to use.
  }];

  let parameters = (ins
    DefaultValuedParameter<"unsigned", "2">:$dimension,
    "::mlir::polynomial::RingAttr":$ring
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // HEIR_LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
