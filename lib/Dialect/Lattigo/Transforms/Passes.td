#ifndef LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AllocToInPlace : Pass<"lattigo-alloc-to-inplace"> {
  let summary = "Convert AllocOps to InPlaceOps in Lattigo";
  let description = [{
    This pass converts AllocOps to InPlaceOps in Lattigo.

  }];
  let dependentDialects = ["mlir::heir::lattigo::LattigoDialect"];
}

def ConfigureCryptoContext : Pass<"lattigo-configure-crypto-context"> {
  let summary = "Configure the crypto context in Lattigo";
  let description = [{
    This pass generates helper functions to configure the Lattigo objects for the given function.

    For example, for an MLIR function `@my_func`, the generated helpers have the following signatures
    ```mlir
    func.func @my_func__configure() -> (!lattigo.bgv.evaluator, !lattigo.bgv.parameter, !lattigo.bgv.encoder, !lattigo.rlwe.encryptor, !lattigo.rlwe.decryptor)
    ```
  }];
  let dependentDialects = ["mlir::heir::lattigo::LattigoDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
  ];
}

#endif  // LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_TD_
