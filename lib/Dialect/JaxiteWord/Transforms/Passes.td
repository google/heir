#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"jaxiteword-configure-crypto-context"> {
  let summary = "Configure the crypto context for CROSS/JaxiteWord";
  let description = [{
    This pass generates helper functions to generate and configure the CROSS
    crypto context for the given function. It analyzes the entry function to
    determine the required CKKS scheme parameters, multiplication depth,
    rotation indices, and relinearization needs, then synthesizes setup
    functions using `jaxiteword.gen_params` and
    `jaxiteword.program_initialization`.

    For example, for an MLIR function `@my_func`, the generated helpers have
    the following signatures:
    ```mlir
    func.func @my_func__generate_crypto_context()
        -> !jaxiteword.crypto_context

    func.func @my_func__configure_crypto_context(
        !jaxiteword.crypto_context, !jaxiteword.public_key,
        !jaxiteword.private_key, !jaxiteword.eval_key)
    ```
  }];
  let dependentDialects = ["mlir::heir::jaxiteword::JaxiteWordDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Name of entry function.">,
    Option<"mulDepth", "mul-depth", "int",
           /*default=*/"0", "Manually specify the mul depth (overrides analysis)">,
    Option<"dnum", "dnum", "int",
           /*default=*/"3", "Key-switching decomposition parameter">,
    Option<"r", "r", "int",
           /*default=*/"4", "TPU tiling row dimension (r * c = degree)">,
    Option<"c", "c", "int",
           /*default=*/"4", "TPU tiling column dimension (r * c = degree)">,
    Option<"batch", "batch", "int",
           /*default=*/"1", "TPU batch size">,
    Option<"compositeDegree", "composite-degree", "int",
           /*default=*/"1", "Number of Q moduli per logical rescale step. "
           "Passed through to CROSS for composite rescaling.">,
    Option<"scalingFactor", "scaling-factor", "double",
           /*default=*/"563019763943521.0", "Default scaling factor.">,
  ];
}

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
