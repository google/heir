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

def LWE_BitFieldEncoding: LWE_EncodingAttr<"BitFieldEncoding", "bit_field_encoding"> {
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

  let parameters = (ins
    "unsigned":$cleartext_start,
    "unsigned":$cleartext_bitwidth
  );
}
#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_TD_
