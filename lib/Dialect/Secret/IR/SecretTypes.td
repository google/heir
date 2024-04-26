#ifndef HEIR_LIB_DIALECT_SECRET_IR_SECRETTYPES_TD_
#define HEIR_LIB_DIALECT_SECRET_IR_SECRETTYPES_TD_

include "SecretDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Secret_Type<string name, string typeMnemonic>
    : TypeDef<Secret_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Secret : Secret_Type<"Secret", "secret"> {
  let summary = "A secret value";

  let description = [{
    A generic wrapper around another MLIR type, representing an encrypted value
    but not specifying the manner of encryption. This is useful in HEIR because
    the compiler may choose various details of the FHE scheme based on the
    properties of the input program, the backend target hardware, and cost
    models of the various passes.
  }];

  let parameters = (ins "Type":$valueType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$valueType), [{
      return $_get(valueType.getContext(), valueType);
    }]>
  ];
  let assemblyFormat = "`<` $valueType `>`";
}

#endif  // HEIR_LIB_DIALECT_SECRET_IR_SECRETTYPES_TD_
