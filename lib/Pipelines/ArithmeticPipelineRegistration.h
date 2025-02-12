#ifndef LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_

#include <cstdint>
#include <functional>
#include <string>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

namespace mlir::heir {

// RLWE scheme selector
enum RLWEScheme { ckksScheme, bgvScheme };

struct SimdVectorizerOptions
    : public PassPipelineOptions<SimdVectorizerOptions> {
  PassOptions::Option<bool> experimentalDisableLoopUnroll{
      *this, "experimental-disable-loop-unroll",
      llvm::cl::desc("Experimental: disable loop unroll, may break analyses "
                     "(default to false)"),
      llvm::cl::init(false)};
};

void heirSIMDVectorizerPipelineBuilder(OpPassManager &manager,
                                       bool disableLoopUnroll);

struct MlirToRLWEPipelineOptions : public SimdVectorizerOptions {
  PassOptions::Option<int> ciphertextDegree{
      *this, "ciphertext-degree",
      llvm::cl::desc("The degree of the polynomials to use for ciphertexts; "
                     "equivalently, the number of messages that can be packed "
                     "into a single ciphertext."),
      llvm::cl::init(1024)};
  PassOptions::Option<bool> usePublicKey{
      *this, "use-public-key",
      llvm::cl::desc("If true, generate a client interface that uses a public "
                     "key for encryption."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> modulusSwitchBeforeFirstMul{
      *this, "modulus-switch-before-first-mul",
      llvm::cl::desc("Modulus switching right before the first multiplication "
                     "(default to false)"),
      llvm::cl::init(false)};
  PassOptions::Option<int64_t> plaintextModulus{
      *this, "plaintext-modulus",
      llvm::cl::desc("Plaintext modulus for BGV scheme (default to 65537)"),
      llvm::cl::init(65537)};
  PassOptions::Option<std::string> noiseModel{
      *this, "noise-model",
      llvm::cl::desc("Noise model to use during parameter generation, see "
                     "--validate-noise pass options for available models"
                     "(default to bgv-noise-by-bound-coeff-average-case-pk)"),
      llvm::cl::init("bgv-noise-by-bound-coeff-average-case-pk")};
};

struct BackendOptions : public PassPipelineOptions<BackendOptions> {
  PassOptions::Option<std::string> entryFunction{
      *this, "entry-function", llvm::cl::desc("Entry function"),
      llvm::cl::init("main")};
  PassOptions::Option<bool> debug{
      *this, "insert-debug-handler-calls",
      llvm::cl::desc("Insert function calls to an externally-defined debug "
                     "function (cf. --lwe-add-debug-port)"),
      llvm::cl::init(false)};
};

using RLWEPipelineBuilder =
    std::function<void(OpPassManager &, const MlirToRLWEPipelineOptions &)>;

using BackendPipelineBuilder =
    std::function<void(OpPassManager &, const BackendOptions &)>;

void mlirToRLWEPipeline(OpPassManager &pm,
                        const MlirToRLWEPipelineOptions &options,
                        RLWEScheme scheme);

void mlirToSecretArithmeticPipelineBuilder(
    OpPassManager &pm, const MlirToRLWEPipelineOptions &options);

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(RLWEScheme scheme);

BackendPipelineBuilder toOpenFhePipelineBuilder();

BackendPipelineBuilder toLattigoPipelineBuilder();

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
