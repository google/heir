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

struct MlirToSecretArithmeticPipelineOptions
    : public PassPipelineOptions<MlirToSecretArithmeticPipelineOptions> {
  PassOptions::Option<std::string> entryFunction{
      *this, "entry-function", llvm::cl::desc("Entry function to secretize"),
      llvm::cl::init("main")};
};

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
};

typedef std::function<void(OpPassManager &pm,
                           const MlirToRLWEPipelineOptions &options)>
    RLWEPipelineBuilder;

void mlirToRLWEPipeline(OpPassManager &pm,
                        const MlirToRLWEPipelineOptions &options,
                        RLWEScheme scheme);

void mlirToSecretArithmeticPipelineBuilder(
    OpPassManager &pm, const MlirToSecretArithmeticPipelineOptions &options);

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(RLWEScheme scheme);

RLWEPipelineBuilder mlirToOpenFheRLWEPipelineBuilder(RLWEScheme scheme);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
