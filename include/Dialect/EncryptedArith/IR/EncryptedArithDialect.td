#ifndef THIRD_PARTY_HEIR_INCLUDE_DIALECT_ENCRYPTEDARITH_IR_ENCRYPTEDARITHDIALECT_TD_
#define THIRD_PARTY_HEIR_INCLUDE_DIALECT_ENCRYPTEDARITH_IR_ENCRYPTEDARITHDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// EncryptedArith dialect definitions
//===----------------------------------------------------------------------===//

def EncryptedArith_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "enc_arith";

  let summary = "A dialect for encrypted analogues of builtin types";
  let description = [{
    The EncryptedArith dialect defines high level primitive types for FHE. It
    is intended to be used in a high level representation of an encrypted
    arithmetic program.
  }];

  let cppNamespace = "::mlir::heir";

  let useDefaultTypePrinterParser = 1;
}

#endif  // THIRD_PARTY_HEIR_INCLUDE_DIALECT_ENCRYPTEDARITH_IR_ENCRYPTEDARITHDIALECT_TD_
