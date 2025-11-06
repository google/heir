#ifndef LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
#define LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Below defines new LWE attributes following
// [#785](https://github.com/google/heir/issues/785).

class LWE_OverflowAttr<string attrName, string attrMnemonic>
  : AttrDef<LWE_Dialect, attrName> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def LWE_NoOverflowAttr : LWE_OverflowAttr<"NoOverflow", "no_overflow"> {
  let summary = "An attribute informing that application data never overflows.";
  let description = [{
    This attribute informs lowerings that a program is written so that the message data
    will never overflow beyond the message type.

    // FIXME: Have a separate WraparoundOverflow, which lowers the same as NoOverflow?
  }];
}

def LWE_PreserveOverflowAttr : LWE_OverflowAttr<"PreserveOverflow", "preserve_overflow"> {
  let summary = "An attribute informing that application data overflows in the message type.";
  let description = [{
    This attribute informs lowerings that a program is written so that the message data
    may overflow beyond the message type.
  }];
}

def AnyOverflowAttr : AnyAttrOf<[LWE_NoOverflowAttr, LWE_PreserveOverflowAttr]> {
  string cppType = "Attribute";
}

def LWE_ApplicationDataAttr : AttrDef<LWE_Dialect, "ApplicationData"> {
  let mnemonic = "application_data";
  let description = [{
    An attribute describing the semantics of the underlying application data.

    The `messageType` parameter is used to describe the type and bits of the
    original application data, e.g. i1, i32, f32. This type is later mapped
    into the plaintext space of an FHE scheme by embedding, scaling, or other
    techniques.

    This attribute also contains information about the overflow semantics of the
    data in the application. By default, we assume that the application program
    was written so that the overflow is not expected and the overflow attribute
    can can be `no_overflow`. For LWE-based CGGI ciphertexts, the overflow
    attribute will usually be `preserve_overflow`, since messages will overflow
    into padding bits.
  }];

  let parameters = (ins
    "mlir::Type":$message_type,
    DefaultValuedParameter<"Attribute", "NoOverflowAttr::get($_ctxt)">:$overflow
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$message_type, "Attribute":$overflow), [{
      return $_get(message_type.getContext(), message_type, overflow);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  // Verify that the overflow attribute is one of preserve or no overflow.
  let genVerifyDecl = 1;
}

class LWE_EncodingAttrForLWE<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LWE_Dialect, attrName, traits # [OpAsmAttrInterface]> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # attrMnemonic # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

class LWE_EncodingAttrWithScalingParam<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LWE_EncodingAttrForLWE<attrName, attrMnemonic, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    APIntParameter<"">:$scaling_factor
  );
}

def LWE_ConstantCoefficientEncoding
  : LWE_EncodingAttrWithScalingParam<"ConstantCoefficientEncoding", "constant_coefficient_encoding"> {
  let summary = "An encoding of a scalar in the constant coefficient";
  let description = [{
    An encoding of a single scalar into the constant coefficient of the plaintext.

    All other coefficients of the plaintext are set to be zero. This encoding is
    used to encode scalar LWE ciphertexts where the plaintext space is viewed
    as a polynomial ring modulo `x`.

    The scalar is first multiplied by the `scaling_factor` and then rounded to
    the nearest integer before encoding into the plaintext coefficient.

    Example:

    ```
    #coeff_encoding = #lwe.constant_coefficient_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_CoefficientEncoding
  : LWE_EncodingAttrWithScalingParam<"CoefficientEncoding", "coefficient_encoding"> {
  let summary = "An encoding of cleartexts directly as coefficients.";
  let description = [{
    A coefficient encoding of a list of integers asserts that the coefficients
    of the polynomials contain the integers, with the same semantics as
    `constant_coefficient_encoding` for per-coefficient encodings.

    A `scaling_factor` is optionally applied on the scalar when converting from
    a rounded floating point to an integer.

    Example:

    ```
    #coeff_encoding = #lwe.coefficient_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_InverseCanonicalEmbeddingEncoding
  : LWE_EncodingAttrWithScalingParam<"InverseCanonicalEncoding", "inverse_canonical_encoding"> {
  let summary = "An encoding of cleartexts via the inverse canonical embedding.";
  let description = [{
    Let $n$ be the degree of the polynomials in the plaintext space. An
    "inverse_canonical_encoding" of a list of real or complex values
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

    The scaling parameter $\Delta$ is specified by the `scaling_factor`, which
    are applied coefficient-wise using the same semantics as the
    `constant_coefficient_encoding`.

    A typical flow for the CKKS scheme using this encoding would be to apply an
    inverse FFT operation to invert the canonical embedding to be a polynomial
    with real coefficients, then encrypt scale the resulting polynomial's
    coefficients according to the scaling parameters, then round to get integer
    coefficients.

    Example:

    ```
    #canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_FullCRTPackingEncoding
  : LWE_EncodingAttrWithScalingParam<"FullCRTPackingEncoding", "full_crt_packing_encoding"> {
  let summary = "An encoding of cleartexts via CRT slots.";
  let description = [{
    This encoding maps a list of integers via the Chinese Remainder Theorem (CRT) into the plaintext space.

    Given a ring with irreducible ideal polynomial `f(x)` and coefficient
    modulus `q`, `f(x)` can be decomposed modulo `q` into a direct product of
    lower-degree polynomials. This allows full SIMD-style homomorphic operations
    across the slots formed from each factor.

    This attribute can only be used in the context of on full CRT packing, where
    the polynomial `f(x)` splits completely (into linear factors) and the number
    of slots equals the degree of `f(x)`. This happens when `q` is prime and `q
    = 1 mod n`.

    A `scaling_factor` is optionally applied on the scalar when converting from
    a rounded floating point to an integer.

    Example:

    ```
    #coeff_encoding = #lwe.full_crt_packing_encoding<scaling_factor=10000>
    ```
  }];
}

def AnyPlaintextEncodingAttr : AnyAttrOf<[LWE_ConstantCoefficientEncoding, LWE_CoefficientEncoding, LWE_InverseCanonicalEmbeddingEncoding, LWE_FullCRTPackingEncoding]> {
  string cppType = "Attribute";
}

def LWE_PlaintextSpaceAttr : AttrDef<LWE_Dialect, "PlaintextSpace"> {
  let mnemonic = "plaintext_space";
  let description = [{
    An attribute describing the plaintext space and the transformation from
    application data to plaintext space of an FHE scheme.

    The plaintext space information is the ring structure, which contains the
    plaintext modulus $t$, which may be a power of two in the case of CGGI
    ciphertexts, or a prime power for RLWE. LWE ciphertexts use the
    ideal polynomial of degree 1 $x$. The plaintext modulus used in LWE-based
    CGGI plaintexts describes the full message space $\mathbb{Z}_p$ including
    the padding bits. The application data info attribute describes the space
    $\mathbb{Z}_p'$ where $p' < p$ that the underlying message belongs to.

    For RLWE schemes, this will include the type of encoding of application data
    integers to a plaintext space `Z_p[X]/X^N + 1`. This may be a constant
    coefficient encoding, CRT-based packing for SIMD semantics, or other slot
    packing. When using full CRT packing, the ring must split into linear
    factors. The CKKS scheme will also include attributes describing the complex
    encoding, including the scaling factor, which will change after
    multiplication and rescaling.
  }];

  let parameters = (ins
    "::mlir::heir::polynomial::RingAttr":$ring,
    AnyPlaintextEncodingAttr:$encoding
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

def LWE_KeyAttr : AttrDef<LWE_Dialect, "Key", [OpAsmAttrInterface]> {
  let mnemonic = "key";
  let description = [{
    An attribute describing the key with which the message is currently
    encrypted.

    The key attribute describes the key with which the message is currently
    encrypted and decryption can be performed. For example, if the decryption of
    a ciphertext $c = (c_0(x), c_1(x))$ is performed by computing the inner
    product $(c_0(x), c_1(x)) \cdot (1, s(x))$ then the key is $(1, s(x))$.

    The `slot_index` describes the key after using a Galois automorphism to
    rotate the plaintext slots by `slot_index`. This will correspond to an
    action $\phi_k: x \rightarrow x^k$ for some `k` that depends on the
    structure of the Galois group for the chosen scheme parameters. The
    corresponding key will have a new basis $(1, s(x^(k)))$.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int", "0">:$slot_index
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # mnemonic # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def LWE_EncryptionTypeEnum : I32EnumAttr<"LweEncryptionType", "An enum attribute representing an encryption method", [
  I32EnumAttrCase<"msb", 0>,
  I32EnumAttrCase<"lsb", 1>,
  I32EnumAttrCase<"mix", 2>
]> {
    let cppNamespace = "::mlir::heir::lwe";
}

def LWE_CiphertextSpaceAttr : AttrDef<LWE_Dialect, "CiphertextSpace", [OpAsmAttrInterface]> {
  let mnemonic = "ciphertext_space";
  let description = [{
    An attribute describing the ciphertext space and the transformation from
    plaintext space to ciphertext space of an FHE scheme.

    The ciphertext space information includes the ring attribute, describing the
    space that the ciphertext elements belong to. The ring attribute contains a
    coefficient type attribute that describes the semantics of the coefficient.
    For example, a ring modulo $1 + x^1024$ with coefficients modulo $q =
    298374$ will be described as

    ```
    !cmod = !mod_arith.int<298374 : i64>
    #ring = #polynomial.ring<coefficientType = !cmod, polynomialModulus = <1 + x**1024>>
    #ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = lsb>
    ```

    Ciphertexts using an RNS representation for $q$ will use an RNS type in
    their ring's coefficient type attribute.

    ```
    !limb1 = !mod_arith.int<2251799814045697 : i64>
    !limb2 = !mod_arith.int<65537 : i64>
    #rns_mod = !rns.rns<!limb1, !limb2>
    #ring = #polynomial.ring<coefficientType = #rns_mod, polynomialModulus = <1 + x**1024>>
    #ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = lsb>
    ```

    Scalar LWE ciphertexts (like those used in CGGI) use an ideal polynomial of
    degree 1, $x$. CGGI ciphertexts will typically use a power of two modulus
    and may use a native integer type for its coefficient modulus.

    ```
    #ring = #polynomial.ring<coefficientType = i32, polynomialModulus = <1 + x**1024>>
    #ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = lsb>
    ```

    The ciphertext encoding info is used to describe the way the plaintext data
    is encoded into the ciphertext (in the MSB, LSB, or mixed).

    The `size` parameter is used to describe the number of polynomials
    comprising the ciphertext. This is typically 2 for RLWE ciphertexts that
    are made up of an $(a, b)$ pair and greater than 2 for LWE instances. For
    example, after an RLWE multiplication of two size 2 ciphertexts,
    the ciphertext's size will be 3.
  }];

  let parameters = (ins
    "::mlir::heir::polynomial::RingAttr":$ring,
    "::mlir::heir::lwe::LweEncryptionType":$encryption_type,
    DefaultValuedParameter<"unsigned", "2">:$size
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Helper method for other Type/Attribute containing this Attribute.
    void getAliasSuffix(::llvm::raw_ostream& os) const {
      getRing().getAliasSuffix(os);
      auto size = getSize();
      if (size != 2) {
        os << "_D" << size;
      }
    }

    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # mnemonic # [{";
      getAliasSuffix(os);
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def LWE_ModulusChainAttr : AttrDef<LWE_Dialect, "ModulusChain", [OpAsmAttrInterface]> {
  let mnemonic = "modulus_chain";
  let description = [{
    An attribute describing the elements of the modulus chain of an RLWE scheme.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::IntegerAttr">:$elements,
    "int":$current
  );

  let assemblyFormat = "`<` `elements` `=` `<` $elements `>``,` `current` `=` $current `>`";

  // let genVerifyDecl = 1; // Verify index into list

  let extraClassDeclaration = [{
    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # mnemonic # [{";
      os << "_L" << getElements().size() - 1;
      os << "_C" << getCurrent();
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

#endif  // LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
