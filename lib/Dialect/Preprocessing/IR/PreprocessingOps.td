#ifndef LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGOPS_TD_
#define LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGOPS_TD_

include "PreprocessingDialect.td"
include "PreprocessingTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Preprocessing_Op<string mnemonic, list<Trait> traits = []> :
        Op<Preprocessing_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::preprocessing";
}

def Preprocessing_EmptyOp : Preprocessing_Op<"empty"> {
  let summary = "Allocate an empty preprocessing storage";
  let description = [{
    This op creates a new `preprocessing.storage` value with an undetermined size.
  }];
  let results = (outs Res<PreprocessingStorage, "", [MemAlloc]>:$storage);
  let assemblyFormat = "attr-dict `:` type($storage)";
}

def Preprocessing_StoreOp : Preprocessing_Op<"store"> {
  let summary = "Store a value to preprocessing storage at given indices and site_id";
  let description = [{
    This op stores a value into a `preprocessing.storage`, while tracking:

    1. The type being stored.
    2. The `site_id` it came from, which corresponds to an encode op the value is
       used in.
    3. An (optional) multiset of indices corresponding to a particular iteration of
       a loop nest.

    In `preprocessing-to-memref`, this op lowers to storing a plaintext in a
    memref.
  }];

  let arguments = (ins
    AnyType:$value,
    Arg<PreprocessingStorage, "", [MemWrite]>:$storage,
    Variadic<Index>:$indices,
    I32Attr:$site_id,
    TypeAttr:$element_type
  );
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $storage `[` $indices `]` `site` $site_id `<` $element_type `>` attr-dict `:` type($value) `,` type($storage)";
}

def Preprocessing_LoadOp : Preprocessing_Op<"load"> {
  let summary = "Load a value from preprocessing storage at given indices and site_id";
  let description = [{
    This op loads a value from a `preprocessing.storage`, while tracking:

    1. The type being stored.
    2. The `site_id` it came from, which corresponds to an encode op the value is
       used in.
    3. An (optional) multiset of indices corresponding to a particular iteration of
       a loop nest.

    In `preprocessing-to-memref`, this op lowers to extracting a plaintext from
    a memref.
  }];
  let arguments = (ins
    Arg<PreprocessingStorage, "", [MemRead]>:$storage,
    Variadic<Index>:$indices,
    I32Attr:$site_id,
    TypeAttr:$element_type
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$storage `[` $indices `]` `site` $site_id `<` $element_type `>` attr-dict `:` type($storage) `,` type($result)";
}


#endif  // LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGOPS_TD_
