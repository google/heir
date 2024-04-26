#ifndef HEIR_LIB_DIALECT_LWE_IR_LWEDIALECT_TD_
#define HEIR_LIB_DIALECT_LWE_IR_LWEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def LWE_Dialect : Dialect {
  let name = "lwe";
  let description = [{
    The `lwe` dialect is a dialect for concepts related to cryptosystems
    in the Learning With Errors (LWE) family.

    See [Wikipedia](https://en.wikipedia.org/wiki/Learning_with_errors)
    for an overview of LWE and the related
    [RLWE](https://en.wikipedia.org/wiki/Ring_learning_with_errors)
    problem.

    While one might expect this dialect to contain types along the lines
    of LWE and RLWE ciphertexts, and operations like encryption, decryption,
    adding and multiplying ciphertexts, these concepts are not centralized
    here because they are too scheme-specific.

    Instead, this dialect provides attributes that can be attached to tensors
    of integer or `poly.poly` types, which indicate that they are semantically
    LWE and RLWE ciphertexts, respectively.
  }];

  let cppNamespace = "::mlir::heir::lwe";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // HEIR_LIB_DIALECT_LWE_IR_LWEDIALECT_TD_
