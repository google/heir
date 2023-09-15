#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_

include "LWEDialect.td"

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
  let parameters = (ins
    // These two parameters identify a scaling factor to scale cleartexts in
    // preparation for noise growth of FHE schemes. In this format they are
    // restricted to being powers of two.
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

    In LWE the plaintexts are integers of a given bit width, and the cleartext
    messages are integers of a smaller bit width. A common example might be
    3-bit cleartexts inside a 32-bit plaintext. In the CGGI FHE scheme, the
    3-bit cleartext might be stored as follows, where 0 denotes a 0 bit, `b`
    denotes a bit of the cleartext, `n` denotes a bit reserved for noise, and
    `|` is a visual aid to show where the bit fields begin and end.


    ```
       0|bbb|nn...n
    MSB^          ^LSB
    ```

    The data describing the encoding consists of the starting bit positions of
    the cleartext bit field and its width, where the LSB is bit 0 and the MSB
    is bit `bit_width-1`. So the above example would have starting bit `30` and
    width `3`. The bits preceding (more significant than) the starting bit are
    reserved for padding, and the bits following (less significant than) the
    ending bit are reserved for noise.

    The presence of this attribute as the `encoding` attribute of a tensor
    indicates that the tensor is an LWE ciphertext.

    Example:

    ```
    #lwe_encoding = #lwe.poly_coefficient_encoding<cleartext_start=30, cleartext_bitwidth=3>
    %lwe_ciphertext = arith.constant <[1,2,3,4]> : tensor<4xi32, #lwe_encoding>
    ```
  }];
}

def RLWE_PolyCoefficientEncoding
  : LWE_EncodingAttrWithScalingFactor<"PolyCoefficientEncoding", "poly_coefficient_encoding"> {
  let summary = "An attribute describing encoded RLWE plaintexts via coefficients.";
  let description = [{
    A coefficient encoding of a list of integers asserts that the coefficients
    of the polynomials contain the cleartexts, with the same semantics as
    `bit_field_encoding` for per-coefficient encodings.

    The presence of this attribute as the `encoding` attribute of a tensor of
    `poly.poly` indicates that the tensor is an RLWE ciphertext for some RLWE
    scheme that supports the coefficient encoding.

    Example:

    ```
    #generator = #poly.polynomial<1 + x**1024>
    #ring = #poly.ring<cmod=65536, ideal=#generator>
    #coeff_encoding = #lwe.poly_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>

    %poly1 = poly.from_tensor %coeffs1 : tensor<10xi16> -> !poly.poly<#ring>
    %poly2 = poly.from_tensor %coeffs2 : tensor<10xi16> -> !poly.poly<#ring>
    %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring>, #coeff_encoding>
    ```
  }];
}

def RLWE_PolyEvaluationEncoding
  : LWE_EncodingAttrWithScalingFactor<"PolyEvaluationEncoding", "poly_evaluation_encoding"> {
  let summary = "An attribute describing encoded RLWE plaintexts via evaluations at fixed points.";
  let description = [{
    A "evaluation encoding" of a list of integers $(v_1, \dots, v_n)$ asserts
    that $f(\x_1) = v_1, \dots, f(x_n) = v_n$ for some implicit, but fixed and
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
    in the relevant coefficient domain, which allows one to use NTT/INTT to
    tansform quickly between the coefficient and evaluation forms.

    Example:

    ```
    #generator = #poly.polynomial<1 + x**1024>
    #ring = #poly.ring<cmod=65536, ideal=#generator>
    #lwe_encoding = #lwe.poly_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

    %evals = arith.constant <[1, 2, 4, 5]> : tensor<4xi16>
    // Note no `intt` operation exists in poly yet.
    %poly1 = poly.intt %evals : tensor<4xi16> -> !poly.poly<#ring, #eval_encoding>
    %poly2 = poly.intt %evals : tensor<4xi16> -> !poly.poly<#ring, #eval_encoding>
    %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring, #eval_encoding>>
    ```
  }];
}

def RLWE_PolyRoundedEmbeddingEncoding
  : LWE_EncodingAttrWithScalingFactor<"PolyRoundedEmbeddingEncoding", "poly_rounded_embedding_encoding"> {
  let summary = "An attribute describing encoded RLWE plaintexts via the rounded inverse canonical embedding.";
  let description = [{
    Let $n$ be the degree of the polynomials in the plaintext space.
    A "rounded embedding encoding" of a list of real or complex values $v_1,
    \dots, v_{n/2}$ is (almost) the inverse of the following decoding map.

    Map a polynomial $p \in \mathbb{Z}[x] / (x^N + 1) \to \mathbb{C}^{N/2}$
    by evaluating it at the following $N/2$ points, where
    $\omega = e^{2 \pi i / 2N}$ is the primitive $2N$th root of unity:

    \[
      \omega, \omega^3, \omega^5, \dots, \omega^{N-1}
    \]

    This map is called the "canonical embedding" $\tau$, though conjugates are
    discarded for efficiency because $p(\overline{z}) = \overline{p(z)}$.

    Then the complete decoding operation is
    $\textup{Decode}(p) = (1/\Delta)\tau(p)$, where $\Delta$ is a scaling
    parameter.

    The encoding operation is the inverse of the decoding operation, but
    because inputs may be any complex numbers, a rounding step is required
    to invert $\tau$. There are multiple rounding mechanisms, and this
    attribute does not specify which is used, because in theory two ciphertexts
    that have used different roundings are still compatible, though they may
    have different noise growth patterns.

    // TODO: does it make sense to use the bit-field parameters here? It may
    // not work sensibly because the rounding happens after scaling-post-INTT,
    // though you can still use the parameters in this attribute to infer the
    // right power of two to use.
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
    INTT operation to invert the canonical embedding to be a polynomial with
    real coefficients, then encrypt scale the resulting polynomial's
    coefficients according to the scaling parameters, then round to get integer
    coefficients.

    Example:

    ```
    #generator = #poly.polynomial<1 + x**1024>
    #ring = #poly.ring<cmod=65536, ideal=#generator>
    #lwe_encoding = #lwe.poly_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

    %evals = arith.constant <[1, 2, 4, 5]> : tensor<4xi16>
    // Note no `intt` operation exists in poly yet.
    %poly1 = poly.intt %evals : tensor<4xi16> -> !poly.poly<#ring, #eval_encoding>
    %poly2 = poly.intt %evals : tensor<4xi16> -> !poly.poly<#ring, #eval_encoding>
    %rlwe_ciphertext = tensor.from_elements %poly1, %poly2 : tensor<2x!poly.poly<#ring, #eval_encoding>>
    ```
  }];
}

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
