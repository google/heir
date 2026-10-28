#include <cmath>
#include <cstdint>
#include <optional>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/RangeAnalysis/RangeAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Parameters/Cheddar/ParameterFile.h"
#include "lib/Utils/LogArithmetic.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "generate-param-ckks"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMCKKS
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

namespace {
bool containsBootstrap(Operation* op) {
  auto result = op->walk([&](Operation* walkOp) {
    if (isa<ResetsMulDepthOpInterface>(walkOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}
}  // namespace

struct GenerateParamCKKS : impl::GenerateParamCKKSBase<GenerateParamCKKS> {
  using GenerateParamCKKSBase::GenerateParamCKKSBase;

  // In CKKS, the modulus for L0 should be larger than the
  // scaling modulus, however, the number of extra bits is often
  // empirically chosen. We use RangeAnalysis to find the
  // maximum number of extra bits needed for the L0 modulus.
  // TODO(#2754): improve this analysis
  std::optional<int> getExtraBitsForLevel0() {
    LDBG() << "Using range analysis to determine extra bits for level 0";
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    // RangeAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // For double input in range [-1, 1], we use Log2Arithmetic::of(1) to
    // represent it.
    solver.load<RangeAnalysis>(Log2Arithmetic::of(inputRange));
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    std::optional<double> extraBits;

    getOperation()->walk([&](Operation* op) {
      for (auto result : op->getResults()) {
        if (mgmt::shouldHaveMgmtAttribute(result, &solver) &&
            getLevelFromMgmtAttr(result) == 0) {
          auto range = getRange(result, &solver);
          if (range.has_value()) {
            auto resultExtraBits = range->getLog2Value();
            if (!extraBits.has_value() || resultExtraBits > extraBits.value()) {
              extraBits = resultExtraBits;
            }
          }
        }
      }
    });

    if (!extraBits.has_value()) {
      return std::nullopt;
    }
    // 2 more bits for cushion
    int level0ModBits = ceil(extraBits.value()) + 2;
    LDBG() << "Decided on " << level0ModBits << " bits for level 0";
    return level0ModBits;
  }

  // Records the slot count the program asked for and the slot count the ring
  // actually provides (N/2) as module attributes.
  void setSlotCountAttrs(int logN) {
    OpBuilder builder(&getContext());
    getOperation()->setAttr(kRequestedSlotCountAttrName,
                            builder.getI64IntegerAttr(minSlotCount));
    getOperation()->setAttr(kActualSlotCountAttrName,
                            builder.getI64IntegerAttr(1LL << (logN - 1)));
  }

  void runOnOperation() override {
    LDBG() << "Starting generate-param-ckks pass";

    std::optional<int> maxLevel = getMaxLevel(getOperation());
    LDBG() << "Max level identified as " << maxLevel;

    if (!parameterFile.empty()) {
      for (StringRef existingAttrName :
           {StringRef(ckks::CKKSDialect::kSchemeParamAttrName),
            StringRef(cheddar::kParameterSetAttrName)}) {
        if (getOperation()->hasAttr(existingAttrName)) {
          getOperation()->emitOpError()
              << "cannot specify parameter-file when module already has "
              << existingAttrName << " attribute";
          signalPassFailure();
          return;
        }
      }
      if (moduleIsCheddar(getOperation())) {
        auto loadedParamSet = cheddar::loadParameterSetFromFile(
            &getContext(), parameterFile,
            [&]() { return getOperation()->emitError(); });
        if (failed(loadedParamSet)) {
          signalPassFailure();
          return;
        }
        if (failed(cheddar::populateSchemeParamFromParameterSet(
                cast<ModuleOp>(getOperation()), *loadedParamSet))) {
          signalPassFailure();
          return;
        }
        setSlotCountAttrs(loadedParamSet->getLogN());
      } else {
        auto loadedAttr = ckks::loadSchemeParamAttrFromFile(
            &getContext(), parameterFile,
            [&]() { return getOperation()->emitError(); });
        if (failed(loadedAttr)) {
          signalPassFailure();
          return;
        }
        setSlotCountAttrs(loadedAttr->getLogN());
        getOperation()->setAttr(ckks::CKKSDialect::kSchemeParamAttrName,
                                *loadedAttr);
      }
    } else if (auto paramSetAttr =
                   getOperation()->getAttrOfType<cheddar::ParameterSetAttr>(
                       cheddar::kParameterSetAttrName)) {
      if (auto existingSchemeParam =
              getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                  ckks::CKKSDialect::kSchemeParamAttrName)) {
        // Both attributes are present: make sure they agree.
        auto derivedSchemeParam = cheddar::getSchemeParamAttrFromParameterSet(
            paramSetAttr, [&]() { return getOperation()->emitError(); });
        if (failed(derivedSchemeParam)) {
          signalPassFailure();
          return;
        }
        if (*derivedSchemeParam != existingSchemeParam) {
          getOperation()->emitOpError()
              << ckks::CKKSDialect::kSchemeParamAttrName
              << " attribute is inconsistent with "
              << cheddar::kParameterSetAttrName << " attribute: expected "
              << *derivedSchemeParam << " but found " << existingSchemeParam;
          signalPassFailure();
          return;
        }
      } else {
        if (failed(cheddar::populateSchemeParamFromParameterSet(
                cast<ModuleOp>(getOperation()), paramSetAttr))) {
          signalPassFailure();
          return;
        }
        setSlotCountAttrs(paramSetAttr.getLogN());
      }
    }

    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                ckks::CKKSDialect::kSchemeParamAttrName)) {
      // TODO: put this in validate-noise once CKKS noise model is in
      auto schemeParam = ckks::getSchemeParamFromAttr(schemeParamAttr);
      if (schemeParam.getLevel() < maxLevel.value_or(0)) {
        getOperation()->emitOpError()
            << "The level in the scheme param is smaller than the max level.\n";
        signalPassFailure();
        return;
      }
      // In CHEDDAR, fresh ciphertexts are encrypted at defaultEncryptionLevel;
      // the levels above it are reserved for bootstrapping and must not be
      // consumed by the program itself.
      if (auto paramSetAttr =
              getOperation()->getAttrOfType<cheddar::ParameterSetAttr>(
                  cheddar::kParameterSetAttrName);
          paramSetAttr &&
          paramSetAttr.getDefaultEncryptionLevel().has_value() &&
          *paramSetAttr.getDefaultEncryptionLevel() < maxLevel.value_or(0)) {
        getOperation()->emitOpError()
            << "The max level (" << maxLevel.value_or(0)
            << ") exceeds the Cheddar defaultEncryptionLevel ("
            << *paramSetAttr.getDefaultEncryptionLevel()
            << "); levels above it are reserved for bootstrapping.\n";
        signalPassFailure();
        return;
      }
      if (schemeParam.getRingDim() / 2 < minSlotCount) {
        getOperation()->emitOpError()
            << "The slot count in the scheme param ("
            << (schemeParam.getRingDim() / 2)
            << ") is smaller than the minimum slot count ("
            << static_cast<int>(minSlotCount) << ").\n";
        signalPassFailure();
        return;
      }
      return;
    }

    if (firstModBits == 0 || validateFirstModBits) {
      auto extraBits = getExtraBitsForLevel0();
      if (!extraBits.has_value()) {
        emitError(getOperation()->getLoc())
            << "Cannot generate CKKS parameters without first modulus bits "
               "or extra bits for level 0.\n";
        signalPassFailure();
        return;
      }

      if (firstModBits == 0) {
        firstModBits = scalingModBits + extraBits.value();
        LDBG() << "First modulus bits not specified, using " << firstModBits
               << " bits.";
      } else if (extraBits.has_value() &&
                 firstModBits - scalingModBits < extraBits.value()) {
        emitWarning(getOperation()->getLoc())
            << "Range Analysis indicate that the first modulus must be larger "
               "than the scaling modulus by at least "
            << extraBits.value() << " bits.\n";
      }
    }
    LDBG() << "First modulus finalized as having " << firstModBits << " bits";

    // The data occupies minSlotCount slots regardless of how large the ring
    // has to be, so the layouts' packing width is recorded before any bump
    // below. Widening it would desync the packed layouts from the ciphertexts.
    int64_t requestedSlotCount = minSlotCount;

    // for lattigo, defaults to extended encryption technique
    if (moduleIsLattigo(getOperation())) {
      encryptionTechniqueExtended = true;
      LDBG() << "For lattigo, fixing extended encryption technique";

      // Lattigo bootstrapping requires LogN >= 14, i.e., ringDim >= 16384.
      // Since ringDim is computed from minSlotCount (minRingDim = 2 *
      // minSlotCount), we bump minSlotCount to 8192 if bootstrapping is
      // present.
      if (containsBootstrap(getOperation())) {
        if (minSlotCount < 8192) {
          LDBG() << "Lattigo bootstrapping detected, bumping minSlotCount from "
                 << minSlotCount << " to 8192";
          minSlotCount = 8192;
        }
      }
    }

    auto schemeParam = ckks::SchemeParam::getConcreteSchemeParam(
        firstModBits, scalingModBits, maxLevel.value_or(0), minSlotCount,
        usePublicKey, encryptionTechniqueExtended, reducedError);

    LDBG() << "Scheme Param:\n" << schemeParam;

    auto* context = &getContext();
    OpBuilder builder(context);
    getOperation()->setAttr(kRequestedSlotCountAttrName,
                            builder.getI64IntegerAttr(requestedSlotCount));
    getOperation()->setAttr(
        kActualSlotCountAttrName,
        builder.getI64IntegerAttr(schemeParam.getRingDim() / 2));

    // annotate ckks::SchemeParamAttr to ModuleOp
    getOperation()->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            context, log2(schemeParam.getRingDim()),
            DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getPi())),
            schemeParam.getLogDefaultScale(),
            usePublicKey ? ckks::CKKSEncryptionType::pk
                         : ckks::CKKSEncryptionType::sk,
            encryptionTechniqueExtended
                ? ckks::CKKSEncryptionTechnique::extended
                : ckks::CKKSEncryptionTechnique::standard,
            /*moduliPerLevel=*/nullptr));
  }
};

}  // namespace heir
}  // namespace mlir
