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
enum RLWEScheme { ckksScheme, bgvScheme, bfvScheme };

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
      llvm::cl::desc("If true, use public key encryption (default to true)"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> encryptionTechniqueExtended{
      *this, "encryption-technique-extended",
      llvm::cl::desc("If true, use extended encryption technique (default to "
                     "false)"),
      llvm::cl::init(false)};
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
                     "--generate-param pass options for available models"),
      llvm::cl::init("")};
  PassOptions::Option<bool> annotateNoiseBound{
      *this, "annotate-noise-bound",
      llvm::cl::desc("If true, the noise predicted by noise model is annotated "
                     "in the IR."),
      llvm::cl::init(false)};
  PassOptions::Option<int> firstModBits{
      *this, "first-mod-bits",
      llvm::cl::desc("The number of bits in the first modulus for CKKS"),
      llvm::cl::init(55)};
  PassOptions::Option<int> scalingModBits{
      *this, "scaling-mod-bits",
      llvm::cl::desc("The number of bits in the scaling modulus for CKKS"),
      llvm::cl::init(45)};
  PassOptions::Option<int> bfvModBits{
      *this, "bfv-mod-bits",
      llvm::cl::desc("The number of bits for all moduli for B/FV"),
      llvm::cl::init(60)};
  PassOptions::Option<int> ckksBootstrapWaterline{
      *this, "ckks-bootstrap-waterline",
      llvm::cl::desc("The number of levels to keep until bootstrapping in CKKS "
                     "(c.f. --secret-insert-mgmt-ckks)"),
      llvm::cl::init(10)};
  PassOptions::Option<std::string> plaintextExecutionResultFileName{
      *this, "plaintext-execution-result-file-name",
      llvm::cl::desc("File name to import execution result from (c.f. --secret-"
                     "import-execution-result)"),
      llvm::cl::init("")};
};

struct PlaintextBackendOptions
    : public PassPipelineOptions<PlaintextBackendOptions> {
  PassOptions::Option<int64_t> plaintextModulus{
      *this, "plaintext-modulus",
      llvm::cl::desc("Plaintext modulus for BGV/BFV scheme (if not specified, "
                     "execute in the original integer type)"),
      llvm::cl::init(0)};
  PassOptions::Option<int64_t> logScale{
      *this, "log-scale",
      llvm::cl::desc(
          "Log base 2 of the scale for encoding floating points as ints."),
      llvm::cl::init(0)};
  PassOptions::Option<bool> debug{
      *this, "insert-debug-handler-calls",
      llvm::cl::desc("Insert function calls to an externally-defined debug "
                     "function (cf. --secret-add-debug-port)"),
      llvm::cl::init(false)};
  PassOptions::Option<int> plaintextSize{
      *this, "plaintext-size",
      llvm::cl::desc("The size of the plaintexts; i.e., the number of slots "
                     "to use for plaintext packing."),
      llvm::cl::init(1024)};
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

void mlirToPlaintextPipelineBuilder(OpPassManager &pm,
                                    const PlaintextBackendOptions &options);

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(RLWEScheme scheme);

BackendPipelineBuilder toOpenFhePipelineBuilder();

BackendPipelineBuilder toLattigoPipelineBuilder();

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
