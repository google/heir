#ifndef LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_

#include <functional>
#include <string>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

namespace mlir::heir {

// RLWE scheme selector
enum RLWEScheme { ckksScheme, bgvScheme };

void heirSIMDVectorizerPipelineBuilder(OpPassManager &manager);

struct MlirToRLWEPipelineOptions
    : public PassPipelineOptions<MlirToRLWEPipelineOptions> {
  PassOptions::Option<std::string> entryFunction{
      *this, "entry-function", llvm::cl::desc("Entry function to secretize"),
      llvm::cl::init("main")};
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
  PassOptions::Option<bool> oneValuePerHelperFn{
      *this, "one-value-per-helper-fn",
      llvm::cl::desc(
          "If true, split encryption helpers into separate functions "
          "for each SSA value."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> modulusSwitchBeforeFirstMul{
      *this, "modulus-switch-before-first-mul",
      llvm::cl::desc("Modulus switching right before the first multiplication "
                     "(default to false)"),
      llvm::cl::init(false)};
};

using RLWEPipelineBuilder =
    std::function<void(OpPassManager &, const MlirToRLWEPipelineOptions &)>;

void mlirToRLWEPipeline(OpPassManager &pm,
                        const MlirToRLWEPipelineOptions &options,
                        RLWEScheme scheme);

void mlirToSecretArithmeticPipelineBuilder(OpPassManager &pm);

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(RLWEScheme scheme);

RLWEPipelineBuilder mlirToOpenFheRLWEPipelineBuilder(RLWEScheme scheme);

RLWEPipelineBuilder mlirToLattigoRLWEPipelineBuilder(RLWEScheme scheme);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
