#ifndef LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAMCOMMON_H_
#define LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAMCOMMON_H_

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/NoiseAnalysis/NoiseAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/BGV/Params.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

inline void annotateSchemeParam(Operation* op,
                                const bgv::SchemeParam& schemeParam,
                                int64_t minSlotCount, bool usePublicKey,
                                bool encryptionTechniqueExtended) {
  MLIRContext* context = op->getContext();
  OpBuilder builder(context);
  op->setAttr(kRequestedSlotCountAttrName,
              builder.getI64IntegerAttr(minSlotCount));
  op->setAttr(kActualSlotCountAttrName,
              builder.getI64IntegerAttr(schemeParam.getRingDim()));

  op->setAttr(
      bgv::BGVDialect::kSchemeParamAttrName,
      bgv::SchemeParamAttr::get(
          context, log2(schemeParam.getRingDim()),
          DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getQi())),
          DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getPi())),
          schemeParam.getPlaintextModulus(),
          usePublicKey ? bgv::BGVEncryptionType::pk
                       : bgv::BGVEncryptionType::sk,
          encryptionTechniqueExtended ? bgv::BGVEncryptionTechnique::extended
                                      : bgv::BGVEncryptionTechnique::standard));
}

inline void generateFallbackParam(Operation* op, int64_t minSlotCount,
                                  int64_t plaintextModulus, bool usePublicKey,
                                  bool encryptionTechniqueExtended,
                                  double primeBitSize) {
  std::optional<int> maxLevel = getMaxLevel(op);
  std::vector<double> logPrimes(maxLevel.value_or(0) + 1, primeBitSize);

  auto schemeParam = bgv::SchemeParam::getConcreteSchemeParam(
      logPrimes, plaintextModulus, minSlotCount, usePublicKey,
      encryptionTechniqueExtended);

  annotateSchemeParam(op, schemeParam, minSlotCount, usePublicKey,
                      encryptionTechniqueExtended);
}

template <typename NoiseAnalysis>
struct NoiseBoundHelper {
  using NoiseLatticeType = typename NoiseAnalysis::LatticeType;
  using LocalParamType = typename NoiseAnalysis::LocalParamType;

  const typename NoiseAnalysis::SchemeParamType& schemeParam;
  const typename NoiseAnalysis::NoiseModel& noiseModel;
  DataFlowSolver* solver;

  LocalParamType getLocalParam(Value value) const {
    auto level = getLevelFromMgmtAttr(value);
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level.getInt(), dimension);
  }

  double getBound(Value value) const {
    auto localParam = getLocalParam(value);
    auto* noiseLattice = solver->lookupState<NoiseLatticeType>(value);
    return noiseModel.toLogBound(localParam, noiseLattice->getValue());
  }
};

template <typename NoiseModel>
LogicalResult runNoiseAnalysis(
    Operation* op, const typename NoiseModel::SchemeParamType& schemeParam,
    const NoiseModel& model, DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<NoiseAnalysis<NoiseModel>>(schemeParam, model);
  return solver.initializeAndRun(op);
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_GENERATEPARAM_GENERATEPARAMCOMMON_H_
