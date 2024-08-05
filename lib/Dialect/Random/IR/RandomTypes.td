#ifndef LIB_DIALECT_RANDOM_IR_RANDOMTYPES_TD_
#define LIB_DIALECT_RANDOM_IR_RANDOMTYPES_TD_

include "RandomDialect.td"
include "RandomEnums.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"

// A base class for all types in this dialect
class Random_Type<string name, string typeMnemonic>
    : TypeDef<Random_Dialect, name> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def Distribution : Random_Type<"Distribution", "distribution"> {
  let summary = "A random distribution type";

  let description = [{
    A generic type, representing a specific random distribution type of either
    uniform or gaussian as an attribute ($distribution_type).
  }];

  let parameters = (ins "::mlir::heir::random::Distribution":$distribution_type);
}

def PRNG : Random_Type<"PRNG", "prng"> {
  let summary = "A pseudorandom number generator type";

  let description = [{
    A type that provides pseudorandom number generator.
  }];
  let assemblyFormat = "";
}

#endif  // LIB_DIALECT_RANDOM_IR_RANDOMTYPES_TD_
