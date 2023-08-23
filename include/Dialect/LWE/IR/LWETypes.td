#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWETYPES_TD_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWETYPES_TD_

include "LWEDialect.td"
include "LWEAttributes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LWE_Type<string name, string typeMnemonic> : TypeDef<LWE_Dialect, name> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def Plaintext : LWE_Type<"Plaintext", "plaintext"> {
  let summary = "A type for an encoded LWE plaintext";
  let description = [{
    `lwe.plaintext` represents a plaintext value encoding a cleartext.

    Example:

    ```
    #encoding = #lwe.encoding_scheme<
      plaintext_bitwidth=32,
      padding_bitwidth=1,
      cleartext_bitwidth=3>
    %0 = !lwe.plaintext<encoing_scheme=#encoding>
    ```
  }];
  let parameters = (ins LWEEncodingSchemeAttr:$encoding_scheme);
}

// TODO: add an optional noise attribute to track noise
def Ciphertext : LWE_Type<"Ciphertext", "ciphertext"> {
  let summary = "A type for LWE Ciphertext";

  let description = [{
    `lwe.ciphertext` represents an encrypted LWE ciphertext.

    This type tracks the LWE ciphertext parameters, including the ciphertext
    dimension (the number of LWE samples) and the underlying encoding scheme of
    the plaintext. This type lowers to a 1-dimensional tensor of size
    `dimension + 1`, where the extra value is the bias
    $b = \sum_{i=1}^\textup{dimension} a_i s_i$.

    The type is also parameterized by the secret key, as represented by a
    program-unique integer attribute.

    Example:

    ```
    #encoding = #lwe.encoding_scheme<
      plaintext_bitwidth=32,
      padding_bitwidth=1,
      cleartext_bitwidth=3>
    %0 = !lwe.ciphertext<encoing_scheme=#encoding, dimension=1024, secret_key=0>
    ```
  }];

  let parameters = (ins
    LWEEncodingSchemeAttr:$encoding_scheme,
    "unsigned":$dimension,
    // TODO: what else is needed to ensure secret key attributes are unique?
    // e.g., should we have a verifier that ensures any two ciphertexts
    // that have the same secretKey attribute have matching dimensions?
    "unsigned":$secret_key
  );
}

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWETYPES_TD_
