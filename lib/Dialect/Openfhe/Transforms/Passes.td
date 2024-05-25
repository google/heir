#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"openfhe-configure-crypto-context"> {
  let summary = "Configure the crypto context in OpenFHE";
  let description = [{
     This pass generates helper functions to generate and configure the OpenFHE crypto context for the given function. Generating the crypto context sets the appropriate encryption parameters, while the configuration generates the necessary evaluation keys (relinearization and rotation keys).

     For example, for an MLIR function `@my_func`, the generated helpers have the following signatures
     ```mlir
    func.func  @my_func__generate_crypto_context() -> !openfhe.crypto_context

    func.func  @my_func__configure_crypto_context(!openfhe.crypto_context, !openfhe.private_key) -> !openfhe.crypto_context
     ```
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">

  ];
}

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
