#ifndef HEIR_INCLUDE_DIALECT_HEIR_IR_HEIRDIALECT_TD_
#define HEIR_INCLUDE_DIALECT_HEIR_IR_HEIRDIALECT_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"

// ODS Definition for the dialect, see https://mlir.llvm.org/docs/OpDefinitions/
// for more information.

//===----------------------------------------------------------------------===//
// HEIR dialect definitions
//===----------------------------------------------------------------------===//

def HEIR_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "heir";

  let summary = "A high-level dialect for FHE operations";
  let description = [{
    The HEIR dialect defines FHE primitive types and operations. It may defined
    static analysis passes that may annotate programs with data needed
    for optimizations.
  }];

  let cppNamespace = "::mlir::heir";

  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// HEIR dialect types definitions
//===----------------------------------------------------------------------===//

// HEIR dialect's CiphertextType representing an encrypted value.
// The BuildableType allows the type of either an operand or a result to be inferred.
// TODO(asraa): Inherit from the TypeDef class.
def HEIR_Ciphertext : DialectType<HEIR_Dialect, CPred<"$_self.isa<CiphertextType>()">, "ciphertext">,
  BuildableType<"$_builder.getType<::mlir::heir::CiphertextType>()"> {

  let description = [{
    A high-level abstraction of a ciphertext encoding a single value.
  }];

  let summary = "Single-valued Ciphertext";
}


#endif  // HEIR_INCLUDE_DIALECT_HEIR_IR_HEIRDIALECT_TD_
