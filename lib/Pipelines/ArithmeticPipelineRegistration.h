#ifndef LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_

#include <cstdint>
#include <functional>
#include <string>

#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

namespace mlir::heir {

// RLWE scheme selector
enum RLWEScheme { ckksScheme, bgvScheme, bfvScheme };

// Ciphertext management style selector
enum CiphertextManagementStyle { greedy, orbitIlp };

struct LoopOptions : public PassPipelineOptions<LoopOptions> {
  PassOptions::Option<bool> experimentalDisableLoopUnroll{
      *this, "experimental-disable-loop-unroll",
      llvm::cl::desc("Experimental: disable loop unroll, may break analyses "
                     "(default to false)"),
      llvm::cl::init(false)};
};

void hecoSIMDVectorizerPipelineBuilder(OpPassManager& manager,
                                       bool disableLoopUnroll);

struct MlirToRLWEPipelineOptions : public LoopOptions {
  PassOptions::Option<bool> enableArithmetization{
      *this, "enable-arithmetization",
      llvm::cl::desc(
          "If false, skip the arithmetization pipeline and try to directly "
          "lower to RLWE scheme (default to true)"),
      llvm::cl::init(true)};
  PassOptions::Option<int> minSlotCount{
      *this, "min-slot-count",
      llvm::cl::desc("The minimum number of slots needed to pack cleartexts; "
                     "this is a lower bound on the ring degree."),
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
  PassOptions::Option<bool> debug{
      *this, "debug",
      llvm::cl::desc("Insert debug ports after every secret operation."),
      llvm::cl::init(false)};
  PassOptions::Option<std::string> plaintextExecutionResultFileName{
      *this, "plaintext-execution-result-file-name",
      llvm::cl::desc("File name to import execution result from (c.f. --secret-"
                     "import-execution-result)"),
      llvm::cl::init("")};
  PassOptions::Option<bool> enableSplitPreprocessing{
      *this, "enable-split-preprocessing",
      llvm::cl::desc(
          "Split server-side plaintext preprocessing into a separate function"),
      llvm::cl::init(true)};
  PassOptions::Option<CodegenStrategy> codegenStrategy{
      *this, "codegen-strategy",
      llvm::cl::desc("Codegen strategy for assign_layout."),
      llvm::cl::values(
          clEnumValN(CodegenStrategy::AUTO, "auto",
                     "Automatically choose folding based on size"),
          clEnumValN(CodegenStrategy::NEVER_FOLD, "never-fold",
                     "Never fold constants"),
          clEnumValN(CodegenStrategy::FOLD_WHEN_POSSIBLE, "fold-when-possible",
                     "Fold constants when possible")),
      llvm::cl::init(CodegenStrategy::AUTO)};

  // Ciphertext management options
  PassOptions::Option<CiphertextManagementStyle> ciphertextManagementStyle{
      *this, "ciphertext-management-style",
      llvm::cl::desc(
          "Strategy for ciphertext management and bootstrap placement"),
      llvm::cl::values(
          clEnumValN(CiphertextManagementStyle::greedy, "greedy",
                     "Greedy waterline-based management"),
          clEnumValN(CiphertextManagementStyle::orbitIlp, "orbit-ilp",
                     "ILP-based Orbit-style management")),
      llvm::cl::init(CiphertextManagementStyle::greedy)};

  // Greedy-specific options
  PassOptions::Option<bool> greedyModulusSwitchAfterMul{
      *this, "greedy-modulus-switch-after-mul",
      llvm::cl::desc(
          "Modulus switching after the first multiplication (greedy style)"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> greedyModulusSwitchBeforeFirstMul{
      *this, "greedy-modulus-switch-before-first-mul",
      llvm::cl::desc("Modulus switching right before the first multiplication "
                     "(greedy style)"),
      llvm::cl::init(false)};
  PassOptions::Option<int> greedyLevelBudget{
      *this, "greedy-level-budget",
      llvm::cl::desc("Maximum bound on the level for greedy management "
                     "(excluding bootstrap levels)"),
      llvm::cl::init(10)};
  PassOptions::Option<int> greedyBootstrapWaterline{
      *this, "greedy-bootstrap-waterline",
      llvm::cl::desc("The number of remaining levels a ciphertext must have to "
                     "initiate a bootstrap (relative headroom)"),
      llvm::cl::init(0)};

  // Orbit-ILP-specific options
  PassOptions::Option<int> orbitBootstrapWaterline{
      *this, "orbit-bootstrap-waterline",
      llvm::cl::desc("Bootstrap waterline (max level) for Orbit ILP"),
      llvm::cl::init(3)};
  PassOptions::Option<int> orbitScaleWaterline{
      *this, "orbit-scale-waterline",
      llvm::cl::desc("Orbit waterline/base scale bits (Sw)"),
      llvm::cl::init(40)};
  PassOptions::Option<int> orbitScaleFactorBits{
      *this, "orbit-scale-factor-bits",
      llvm::cl::desc("Orbit rescale factor bits (Sf)"), llvm::cl::init(51)};
  PassOptions::Option<int> orbitBootstrapLevelLowerBound{
      *this, "orbit-bootstrap-level-lower-bound",
      llvm::cl::desc("Minimum input level for bootstrap in Orbit constraints"),
      llvm::cl::init(0)};
  PassOptions::Option<std::string> orbitCostModel{
      *this, "orbit-cost-model",
      llvm::cl::desc("Path to JSON cost model for Orbit ILP"),
      llvm::cl::init("")};
  PassOptions::Option<int> orbitBootstrapCost{
      *this, "orbit-bootstrap-cost",
      llvm::cl::desc("Cost of one bootstrap in ILP objective"),
      llvm::cl::init(69320650)};
  PassOptions::Option<int> orbitRescaleCost{
      *this, "orbit-rescale-cost",
      llvm::cl::desc("Cost of one rescale in ILP objective"),
      llvm::cl::init(40988)};
};

struct PlaintextBackendOptions
    : public PassPipelineOptions<PlaintextBackendOptions> {
  PassOptions::Option<int64_t> plaintextModulus{
      *this, "plaintext-modulus",
      llvm::cl::desc("Plaintext modulus for BGV/BFV scheme (if not specified, "
                     "execute in the original integer type)"),
      llvm::cl::init(0)};
  PassOptions::Option<bool> debug{
      *this, "insert-debug-handler-calls",
      llvm::cl::desc("Insert function calls to an externally-defined debug "
                     "function (cf. --secret-add-debug-port)"),
      llvm::cl::init(false)};
  PassOptions::Option<int> plaintextSize{
      *this, "plaintext-size",
      llvm::cl::desc("The size of the plaintexts; i.e., the number of slots "
                     "to use for packing."),
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
  PassOptions::Option<int> firstModSize{
      *this, "first-mod-size",
      llvm::cl::desc("Manually specify the first mod size"), llvm::cl::init(0)};
  PassOptions::Option<int> scalingModSize{
      *this, "scaling-mod-size",
      llvm::cl::desc("Manually specify the scaling mod size"),
      llvm::cl::init(0)};
  PassOptions::Option<int> mulDepth{
      *this, "mul-depth", llvm::cl::desc("Manually specify the mul depth"),
      llvm::cl::init(0)};
  PassOptions::Option<bool> scalingTechniqueFixedManual{
      *this, "scaling-technique-fixed-manual",
      llvm::cl::desc(
          "Whether to use fixed manual scaling technique (defaults to false)"),
      llvm::cl::init(false)};
  PassOptions::Option<int> ringDim{
      *this, "ring-dim", llvm::cl::desc("Manually specify the ring dimension"),
      llvm::cl::init(0)};
  PassOptions::Option<int> batchSize{
      *this, "batch-size", llvm::cl::desc("Manually specify the batch size"),
      llvm::cl::init(0)};
  PassOptions::Option<bool> insecure{
      *this, "insecure", llvm::cl::desc("Whether to use insecure parameter"),
      llvm::cl::init(false)};
};

using RLWEPipelineBuilder =
    std::function<void(OpPassManager&, const MlirToRLWEPipelineOptions&)>;

using BackendPipelineBuilder =
    std::function<void(OpPassManager&, const BackendOptions&)>;

void mlirToRLWEPipeline(OpPassManager& pm,
                        const MlirToRLWEPipelineOptions& options,
                        RLWEScheme scheme);

void mlirToSecretArithmeticPipelineBuilder(
    OpPassManager& pm, const MlirToRLWEPipelineOptions& options);

void mlirToPlaintextPipelineBuilder(OpPassManager& pm,
                                    const PlaintextBackendOptions& options);

RLWEPipelineBuilder mlirToRLWEPipelineBuilder(RLWEScheme scheme);

BackendPipelineBuilder toOpenFhePipelineBuilder();

BackendPipelineBuilder toLattigoPipelineBuilder();

// A subpipeline that preprocesses linalg ops to make them more suitable for
// FHE.
void linalgPreprocessingBuilder(OpPassManager& manager);

void torchLinalgToCkksBuilder(OpPassManager& manager,
                              const MlirToRLWEPipelineOptions& options);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_ARITHMETICPIPELINEREGISTRATION_H_
