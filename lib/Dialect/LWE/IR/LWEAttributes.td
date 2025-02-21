#ifndef LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
#define LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "lib/Dialect/LWE/IR/NewLWEAttributes.td"

class LWE_EncodingAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LWE_Dialect, attrName, traits # [
    // All encoding attributes are required to be compatible with a tensor
    // with an element type relevant to that encoding.
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>,
    OpAsmAttrInterface
]> {
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

class LWE_EncodingAttrWithScalingFactor<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LWE_EncodingAttr<attrName, attrMnemonic, traits> {
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

def AnyPlaintextEncodingInfo : AnyAttrOf<[
  LWE_BitFieldEncoding
]>;

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
    "::mlir::heir::polynomial::RingAttr":$ring
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
