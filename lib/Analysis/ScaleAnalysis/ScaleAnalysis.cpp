#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"

#include <functional>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "ScaleAnalysis"

namespace mlir {
namespace heir {

int64_t BGVScaleModel::evalMulScale(const bgv::LocalParam &param, int64_t lhs,
                                    int64_t rhs) {
  const auto *schemeParam = param.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  return lhs * rhs % t;
}

int64_t BGVScaleModel::evalModReduceScale(const bgv::LocalParam &inputParam,
                                          int64_t scale) {
  const auto *schemeParam = inputParam.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto qi = schemeParam->getQi();
  auto level = inputParam.getCurrentLevel();
  auto qInvT = multiplicativeInverse(APInt(64, qi[level] % t), APInt(64, t));
  return scale * qInvT.getSExtValue() % t;
}

int64_t CKKSScaleModel::evalMulScale(const ckks::LocalParam &param, int64_t lhs,
                                     int64_t rhs) {
  return lhs + rhs;
}

int64_t CKKSScaleModel::evalModReduceScale(const ckks::LocalParam &inputParam,
                                           int64_t scale) {
  const auto *schemeParam = inputParam.getSchemeParam();
  // TODO: rescale using logqi instead of logDefaultScale
  // auto logqi = schemeParam->getLogqi();
  // auto level = inputParam.getCurrentLevel();
  auto logDefaultScale = schemeParam->getLogDefaultScale();
  return scale - logDefaultScale;
}

template <typename ScaleModelT>
LogicalResult ScaleAnalysis<ScaleModelT>::visitOperation(
    Operation *op, ArrayRef<const ScaleLattice *> operands,
    ArrayRef<ScaleLattice *> results) {
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value);
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, const ScaleState &state) {
    auto *lattice = getLatticeElement(value);
    LLVM_DEBUG(llvm::dbgs()
               << "Propagate " << state << " to " << value << "\n");
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  auto getOperandScales = [&](Operation *op, SmallVectorImpl<int64_t> &scales) {
    SmallVector<OpOperand *> secretOperands;
    this->getSecretOperands(op, secretOperands);

    for (auto *operand : secretOperands) {
      auto operandState = getLatticeElement(operand->get())->getValue();
      if (!operandState.isInitialized()) {
        continue;
      }
      scales.push_back(operandState.getScale());
    }
    if (scales.size() > 1) {
      if (scales[0] != scales[1]) {
        LLVM_DEBUG(llvm::dbgs() << "Different scales: " << scales[0] << ", "
                                << scales[1] << " for " << *op << "\n");
      }
    }
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, ScaleState(inputScale));
        }
      })
      .template Case<arith::MulIOp, arith::MulFOp, tensor::ExtractOp>(
          [&](auto mulOp) {
            SmallVector<int64_t> scales;
            getOperandScales(mulOp, scales);
            // there must be at least one secret operand that has scale
            if (scales.empty()) {
              return;
            }
            auto scaleLhs = scales[0];
            auto scaleRhs = scaleLhs;
            // default to the same scale for both operand
            if (scales.size() > 1) {
              scaleRhs = scales[1];
            }

            // NOTE: special case for ExtractOp... it is a mulconst+rotate
            // if not annotated with slot_extract
            // TODO(#1174): decide packing earlier in the pipeline instead of
            // annotation
            if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
              if (extractOp->getAttr("slot_extract")) {
                propagate(mulOp.getResult(), ScaleState(scaleLhs));
                return;
              }
            }

            // propagate scale to result
            auto result = ScaleModelT::evalMulScale(
                getLocalParam(mulOp.getResult()), scaleLhs, scaleRhs);
            propagate(mulOp.getResult(), ScaleState(result));
          })
      .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        SmallVector<int64_t> scales;
        getOperandScales(modReduceOp, scales);
        // there must be at least one secret operand that has scale
        if (scales.empty()) {
          return;
        }

        // propagate scale to result
        auto scale = scales[0];
        // get level of the operand.
        auto newScale = ScaleModelT::evalModReduceScale(
            getLocalParam(modReduceOp.getInput()), scale);

        propagate(modReduceOp.getResult(), ScaleState(newScale));
      })
      .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        // if adjust scale op is not initialized, just do not propagate
        int64_t scale = adjustScaleOp.getScale();
        if (scale < 0) {
          return;
        }
        propagate(adjustScaleOp.getResult(),
                  ScaleState(adjustScaleOp.getScale()));
      })
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        this->getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<int64_t> scales;
        getOperandScales(&op, scales);
        if (scales.empty()) {
          return;
        }

        // just propagate the scale
        for (auto result : secretResults) {
          propagate(result, ScaleState(scales[0]));
        }
      });
  return success();
}

template <typename ScaleModelT>
void ScaleAnalysis<ScaleModelT>::visitExternalCall(
    CallOpInterface call, ArrayRef<const ScaleLattice *> argumentLattices,
    ArrayRef<ScaleLattice *> resultLattices) {
  auto callback = std::bind(&ScaleAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<ScaleState, ScaleLattice>(
      call, argumentLattices, resultLattices, callback);
}

// instantiation
template class ScaleAnalysis<BGVScaleModel>;
template class ScaleAnalysis<CKKSScaleModel>;

}  // namespace heir
}  // namespace mlir
