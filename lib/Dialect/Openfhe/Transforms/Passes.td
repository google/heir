#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"openfhe-configure-crypto-context"> {
  let summary = "Configure the crypto context in OpenFHE";
  let description = [{
    This pass generates helper functions to generate and configure the OpenFHE crypto context for the given function. Generating the crypto context sets the appropriate encryption parameters, while the configuration generates the necessary evaluation keys (relinearization and rotation keys).

    For the options, reader can refer to the OpenFHE documentation at
    https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/README.md#description-of-the-cryptocontext-parameters-and-their-restrictions

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
           "name of entry function.">,
    Option<"mulDepth", "mul-depth", "int",
           /*default=*/"0", "Manually specify the mul depth">,
    // Options for GenParamsOp
    Option<"ringDim", "ring-dim", "int",
           /*default=*/"0", "Manually specify the ring dimension (insecure is implied)">,
    // plaintext modulus is from IR
    Option<"batchSize", "batch-size", "int",
           /*default=*/"0", "Manually specify the batch size">,
    Option<"firstModSize", "first-mod-size", "int",
           /*default=*/"0", "Manually specify the first mod size">,
    Option<"scalingModSize", "scaling-mod-size", "int",
           /*default=*/"0", "Manually specify the scaling mod size">,
    Option<"digitSize", "digit-size", "int",
           /*default=*/"0", "Manually specify the digit size for relinearization">,
    Option<"numLargeDigits", "num-large-digits", "int",
           /*default=*/"0", "Manually specify the number of large digits for HYBRID relinearization">,
    Option<"maxRelinSkDeg", "max-relin-sk-deg", "int",
           /*default=*/"0", "Manually specify the max number of relin sk deg">,
    Option<"insecure", "insecure", "bool",
           /*default=*/"false", "Whether to use insecure parameter (defaults to false)">,
    Option<"keySwitchingTechniqueBV", "key-switching-technique-bv", "bool",
           /*default=*/"false", "Whether to use BV key switching technique (defaults to false)">,
    Option<"scalingTechniqueFixedManual", "scaling-technique-fixed-manual", "bool",
           /*default=*/"false", "Whether to use fixed manual scaling technique (defaults to false)">,
    // For bootstrapping
    Option<"levelBudgetEncode", "level-budget-encode", "int",
           /*default=*/"3", "Level budget for CKKS bootstrap encode (s2c) phase">,
    Option<"levelBudgetDecode", "level-budget-decode", "int",
           /*default=*/"3", "Level budget for CKKS bootstrap decode (c2s) phase">,
  ];
}

def CountAddAndKeySwitch : Pass<"openfhe-count-add-and-key-switch"> {
  let summary = "Count the number of add and key-switch operations in OpenFHE";
  let description = [{
    This pass counts the number of add and key-switch operations in the given function.

    This is used for setting the EvalAddCount and EvalKeySwitchCount in OpenFHE library.
    Cf. [Alexandru et al. 2024](https://eprint.iacr.org/2024/203) for why this
    is important for security.

    The detailed definition of these counts could be found in the KPZ21 paper
    [Revisiting Homomorphic Encryption Schemes for Finite Fields](https://ia.cr/2021/204)

    The pass should be run at the secret arithmetic level when management operations
    have been inserted and the IR is stable.
  }];
}

def FastRotationPrecompute : Pass<"openfhe-fast-rotation-precompute"> {
  let summary = "Identify and apply EvalFastRotation when possible.";
  let description = [{
    This pass identifies when a ciphertext is rotated by multiple different
    shifts, and replaces the `EvalRot` ops with `EvalFastRotationPrecompute`
    followed by `EvalFastRotate`.
  }];
}

def AllocToInPlace : Pass<"openfhe-alloc-to-inplace"> {
  let summary = "Utilize OpenFHE's in-place operations when possible";
  let description = [{
    This pass converts OpenFHE ops that return new ciphertexts to in-place ops.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
