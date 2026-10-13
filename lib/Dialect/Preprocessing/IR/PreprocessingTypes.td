#ifndef LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGTYPES_TD_
#define LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGTYPES_TD_

include "PreprocessingDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

class Preprocessing_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Preprocessing_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PreprocessingStorage : Preprocessing_Type<"PreprocessingStorage", "storage"> {
  let summary = "A storage type wrapping an ordered list of element types";
  let description = [{
    This type is an abstraction over memref (or similar containers)
    holding an ordered list of element types for multi-type storage,
    where the exact layout is not yet decided.

    This allows us to manage server-side plaintext preprocessing as a single
    SSA value, hiding details such as whether there are multiple plaintext types
    that need handling. These details are determined by a later pass that lowers
    `preprocessing.storage` to a backend-specific data structure, often a 1D memref,
    and defers the exact description of that layout until it is easier to determine.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$elementTypes);
  let assemblyFormat = "`<` $elementTypes `>`";
}

def Preprocessing_ResourceDir : Preprocessing_Type<"ResourceDir", "resource_dir"> {
  let summary = "The directory external resources are loaded from";
  let description = [{
    An opaque handle to the directory that `preprocessing.load_resource`
    resolves its relative path against. `preprocessing-use-dynamic-resource-dir`
    adds one as a trailing argument to every function that loads resources, so
    the caller of the public function names the directory at runtime.
  }];
}

#endif  // LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGTYPES_TD_
