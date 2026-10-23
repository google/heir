#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"jaxiteword-configure-crypto-context"> {
  let summary = "Configure the crypto context for CROSS/JaxiteWord";
  let description = [{
    This pass generates helper functions to generate and configure the CROSS
    crypto context for the given function. It analyzes the entry function to
    determine the required CKKS scheme parameters and rotation indices, then
    synthesizes setup functions using `jaxiteword.gen_params` and
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
           /*default=*/"0.0", "Scaling factor override. A positive value "
           "takes precedence over ckks.schemeParam; zero uses "
           "2^logDefaultScale from the scheme parameters.">,
  ];
}

def AnnotateEncodeRescaleLevels
    : Pass<"jaxiteword-annotate-encode-rescale-levels"> {
  let summary = "Annotate plaintext encodes with the ciphertext level whose rescale consumes them";
  let description = [{
    Finds plaintext encodes used by ciphertext-plaintext multiplications whose
    result is immediately modulus-reduced, and records the input ciphertext's
    modulus-chain level on the encode. The JaxiteWord emitter uses this to
    encode plaintext weights at the modulus dropped by the following rescale.
  }];
  let dependentDialects = [
    "mlir::heir::jaxiteword::JaxiteWordDialect",
    "mlir::heir::lwe::LWEDialect"
  ];
}

def OutlineCompiledRegions : Pass<"jaxiteword-outline-compiled-regions"> {
  let summary = "Outline bounded JaxiteWord ciphertext graphs for JAX compilation";
  let description = [{
    Finds connected graphs of raw-array-compatible JaxiteWord operations and
    wraps them in `jaxiteword.compiled_region`. Unsupported ciphertext
    operations form boundaries. Compatible multiply, relinearize, and rescale
    operations remain in the same region as adjacent linear operations.
  }];
  let options = [
    Option<"minimumOperations", "minimum-operations", "int64_t",
           /*default=*/"2", "Minimum number of HE operations in a region">,
    Option<"maximumOperations", "maximum-operations", "int64_t",
           /*default=*/"100000", "Maximum number of HE operations in a region">,
    Option<"maximumRegions", "maximum-regions", "int64_t",
           /*default=*/"64", "Maximum regions outlined per module">
  ];
  let dependentDialects = ["mlir::heir::jaxiteword::JaxiteWordDialect"];
}

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
