#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "ScaleAnalysis"

namespace mlir {
namespace heir {

static bool isAdaptable(Value value) {
  Operation* defOp = value.getDefiningOp();
  if (!defOp) return false;

  if (isa<mgmt::AdjustScaleOp>(defOp)) {
    return true;
  }
  if (isa<mgmt::ModReduceOp, mgmt::RelinearizeOp, mgmt::LevelReduceOp>(defOp)) {
    return isAdaptable(defOp->getOperand(0));
  }
  if (isa<arith::AddIOp, arith::AddFOp>(defOp)) {
    return isAdaptable(defOp->getOperand(0)) &&
           isAdaptable(defOp->getOperand(1));
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ScaleModel
//===----------------------------------------------------------------------===//

int64_t BGVScaleModel::evalAddScale(ArrayRef<int64_t> scales) {
  assert(!scales.empty() && "scales cannot be empty");
  int64_t baseScale = scales[0];
#ifndef NDEBUG
  for (int64_t s : scales) {
    assert(s == baseScale && "BGV scales must match");
  }
#endif
  return baseScale;
}

int64_t BGVScaleModel::evalMulScale(const bgv::LocalParam& param, int64_t lhs,
                                    int64_t rhs) {
  const auto* schemeParam = param.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  return lhs * rhs % t;
}

int64_t BGVScaleModel::evalMulScaleBackward(const bgv::LocalParam& param,
                                            int64_t result, int64_t lhs) {
  const auto* schemeParam = param.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto lhsInv = multiplicativeInverse(APInt(64, lhs), APInt(64, t));
  return result * lhsInv.getSExtValue() % t;
}

int64_t BGVScaleModel::evalModReduceScale(const bgv::LocalParam& inputParam,
                                          int64_t scale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto qi = schemeParam->getQi();
  auto level = inputParam.getCurrentLevel();
  auto qInvT = multiplicativeInverse(APInt(64, qi[level] % t), APInt(64, t));
  return scale * qInvT.getSExtValue() % t;
}

int64_t BGVScaleModel::evalModReduceScaleBackward(
    const bgv::LocalParam& inputParam, int64_t resultScale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto qi = schemeParam->getQi();
  auto level = inputParam.getCurrentLevel();
  return resultScale * (qi[level] % t) % t;
}

std::optional<int64_t> BGVScaleModel::getDefaultScale(
    const bgv::SchemeParam& param) {
  return std::nullopt;
}

int64_t BGVScaleModel::evalMulTargetScale(const bgv::LocalParam& param) {
  assert(false && "BGV should not call evalMulTargetScale");
  return 0;
}

int64_t CKKSScaleModel::evalAddScale(ArrayRef<int64_t> scales) {
  assert(!scales.empty() && "scales cannot be empty");
  int64_t maxScale = scales[0];
  for (int64_t s : scales) {
    maxScale = std::max(maxScale, s);
  }
  return maxScale;
}

int64_t CKKSScaleModel::evalMulScale(const ckks::LocalParam& param, int64_t lhs,
                                     int64_t rhs) {
  // TODO(#1640): support high-precision scale management
  return lhs + rhs;
}

int64_t CKKSScaleModel::evalModReduceScale(const ckks::LocalParam& inputParam,
                                           int64_t scale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  auto level = inputParam.getCurrentLevel();
  const auto& logqi = schemeParam->getLogqi();
  if (level >= 0 && level < static_cast<int>(logqi.size())) {
    return scale - static_cast<int64_t>(std::llround(logqi[level]));
  }
  return scale - schemeParam->getLogDefaultScale();
}

std::optional<int64_t> CKKSScaleModel::getDefaultScale(
    const ckks::SchemeParam& param) {
  return param.getLogDefaultScale();
}

int64_t CKKSScaleModel::evalMulTargetScale(const ckks::LocalParam& param) {
  const auto* schemeParam = param.getSchemeParam();
  auto logDefaultScale = schemeParam->getLogDefaultScale();
  auto level = param.getCurrentLevel();
  const auto& logqi = schemeParam->getLogqi();
  int64_t logqi_level = logDefaultScale;
  if (level >= 0 && level < static_cast<int>(logqi.size())) {
    logqi_level = static_cast<int64_t>(std::llround(logqi[level]));
  }
  return logDefaultScale + logqi_level;
}

//===----------------------------------------------------------------------===//
// ScaleAnalysis (Forward)
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
LogicalResult ScaleAnalysis<ScaleModelT>::visitOperation(
    Operation* op, ArrayRef<const ScaleLattice<ScaleModelT>*> operands,
    ArrayRef<ScaleLattice<ScaleModelT>*> results) {
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, const ScaleState<ScaleModelT>& state) {
    auto* lattice = this->getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Propagate " << state << " to " << value << "\n");
    }
    this->propagateIfChanged(lattice, changed);
  };

  auto getSecretOrInittedOperands =
      [&](Operation* op,
          SmallVectorImpl<OpOperand*>& secretOperands) -> LogicalResult {
    for (auto& opOperand : op->getOpOperands()) {
      std::optional<bool> isSecret =
          this->isSecretInternal(op, opOperand.get());
      if (!isSecret.has_value()) {
        return failure();
      }
      bool isMgmtDefined =
          isa_and_nonnull<mgmt::InitOp>(opOperand.get().getDefiningOp());
      if (*isSecret || isMgmtDefined) {
        secretOperands.push_back(&opOperand);
      }
    }
    return success();
  };

  auto getOperandScales =
      [&](Operation* op, SmallVectorImpl<int64_t>& scales) -> LogicalResult {
    SmallVector<OpOperand*> secretOperands;
    if (failed(getSecretOrInittedOperands(op, secretOperands))) {
      return failure();
    }

    for (auto* operand : secretOperands) {
      auto operandState = this->getLatticeElement(operand->get())->getValue();
      if (!operandState.isInitialized()) {
        if (isAdaptable(operand->get())) {
          continue;
        }
        return failure();
      }
      scales.push_back(operandState.getScale());
    }
    if (scales.size() > 1) {
      if (scales[0] != scales[1]) {
        LLVM_DEBUG(llvm::dbgs() << "Different scales: " << scales[0] << ", "
                                << scales[1] << " for " << *op << "\n");
      }
    }
    return success();
  };

  llvm::TypeSwitch<Operation&>(*op)
      .template Case<arith::MulIOp, arith::MulFOp>([&](auto mulOp) {
        SmallVector<int64_t> scales;
        if (failed(getOperandScales(mulOp, scales))) {
          return;
        }
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

        // propagate scale to result
        int64_t result;
        if (assumeTargetScaleForMul) {
          int64_t targetScale =
              ScaleModelT::evalMulTargetScale(getLocalParam(mulOp.getResult()));
          if (scaleLhs + scaleRhs < targetScale) {
            if (mulOp.getLhs() == mulOp.getRhs()) {
              result = 2 * ((targetScale + 1) / 2);
            } else {
              result = targetScale;
            }
          } else {
            result = scaleLhs + scaleRhs;
          }
        } else {
          result = ScaleModelT::evalMulScale(getLocalParam(mulOp.getResult()),
                                             scaleLhs, scaleRhs);
        }
        propagate(mulOp.getResult(), ScaleState<ScaleModelT>(result));
      })
      .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        SmallVector<int64_t> scales;
        if (failed(getOperandScales(modReduceOp, scales))) {
          return;
        }
        // there must be at least one secret operand that has scale
        if (scales.empty()) {
          return;
        }

        // propagate scale to result
        auto scale = scales[0];
        // get level of the operand.
        auto newScale = ScaleModelT::evalModReduceScale(
            getLocalParam(modReduceOp.getInput()), scale);

        propagate(modReduceOp.getResult(), ScaleState<ScaleModelT>(newScale));
      })
      .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        auto mgmtAttr =
            mgmt::findMgmtAttrAssociatedWith(adjustScaleOp.getResult());
        if (mgmtAttr && mgmtAttr.getScale() != -1) {
          propagate(adjustScaleOp.getResult(),
                    ScaleState<ScaleModelT>(mgmtAttr.getScale()));
        }
        return;
      })
      .template Case<mgmt::InitOp>([&](auto initOp) {
        auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp.getResult());
        // if there is scale annotation, use it
        if (mgmtAttr && mgmtAttr.getScale() != -1) {
          propagate(initOp.getResult(),
                    ScaleState<ScaleModelT>(mgmtAttr.getScale()));
        } else {
          propagate(initOp.getResult(), ScaleState<ScaleModelT>(inputScale));
        }
      })
      .template Case<mgmt::BootstrapOp>([&](auto bootstrapOp) {
        // inputScale is either Delta or Delta^2 depending on the analysis
        // initialization.
        propagate(bootstrapOp.getResult(), ScaleState<ScaleModelT>(inputScale));
      })
      .template Case<arith::AddFOp, arith::SubFOp, arith::AddIOp, arith::SubIOp,
                     tensor::InsertSliceOp, tensor::InsertOp>([&](auto op) {
        SmallVector<int64_t> scales;
        if (failed(getOperandScales(op, scales))) {
          return;
        }
        if (scales.empty()) {
          return;
        }
        auto resultScale = ScaleModelT::evalAddScale(scales);
        for (auto result : op->getResults()) {
          if (this->isSecretInternal(op, result)) {
            propagate(result, ScaleState<ScaleModelT>(resultScale));
          }
        }
      })
      .Default([&](auto& op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        this->getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<int64_t> scales;
        if (failed(getOperandScales(&op, scales))) {
          return;
        }
        if (scales.empty()) {
          return;
        }

        // just propagate the scale
        for (auto result : secretResults) {
          propagate(result, ScaleState<ScaleModelT>(scales[0]));
        }
      });
  return success();
}

template <typename ScaleModelT>
void ScaleAnalysis<ScaleModelT>::visitExternalCall(
    CallOpInterface call,
    ArrayRef<const ScaleLattice<ScaleModelT>*> argumentLattices,
    ArrayRef<ScaleLattice<ScaleModelT>*> resultLattices) {
  auto callback = std::bind(&ScaleAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<ScaleState<ScaleModelT>,
                                  ScaleLattice<ScaleModelT>>(
      call, argumentLattices, resultLattices, callback);
}

// instantiation
template class ScaleAnalysis<BGVScaleModel>;
template class ScaleAnalysis<CKKSScaleModel>;

//===----------------------------------------------------------------------===//
// ScaleAnalysis (Backward)
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
LogicalResult ScaleAnalysisBackward<ScaleModelT>::visitOperation(
    Operation* op, ArrayRef<ScaleLattice<ScaleModelT>*> operands,
    ArrayRef<const ScaleLattice<ScaleModelT>*> results) {
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, const ScaleState<ScaleModelT>& state) {
    auto* lattice = this->getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Back Propagate " << state << " to " << value << "\n");
    }
    this->propagateIfChanged(lattice, changed);
  };

  auto getSecretOrInittedOperands =
      [&](Operation* op,
          SmallVectorImpl<OpOperand*>& secretOperands) -> LogicalResult {
    LLVM_DEBUG(
        { llvm::dbgs() << "secretness of operands for " << *op << ":\n"; });
    for (auto& opOperand : op->getOpOperands()) {
      std::optional<bool> isSecret =
          this->isSecretInternal(op, opOperand.get());
      if (!isSecret.has_value()) {
        return failure();
      }
      bool isMgmtDefined =
          isa_and_nonnull<mgmt::InitOp>(opOperand.get().getDefiningOp());
      LLVM_DEBUG({
        llvm::dbgs() << " " << opOperand.getOperandNumber()
                     << ": isSecret=" << *isSecret
                     << ", isMgmtDefined=" << isMgmtDefined << "\n";
      });
      if (*isSecret || isMgmtDefined) {
        // Treat it as if it were secret for the purpose of scale
        // propagation
        secretOperands.push_back(&opOperand);
      }
    }
    return success();
  };

  auto getOperandScales =
      [&](Operation* op, SmallVectorImpl<int64_t>& operandWithoutScaleIndices,
          SmallVectorImpl<int64_t>& scales) -> LogicalResult {
    LLVM_DEBUG(llvm::dbgs() << "Operand scales for " << op->getName() << ": ");
    SmallVector<OpOperand*> secretOperands;
    if (failed(getSecretOrInittedOperands(op, secretOperands))) {
      return failure();
    }

    for (auto* operand : secretOperands) {
      auto operandState = this->getLatticeElement(operand->get())->getValue();
      if (!operandState.isInitialized()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "o" << operand->getOperandNumber() << "(uninit), ");
        operandWithoutScaleIndices.push_back(operand->getOperandNumber());
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "o" << operand->getOperandNumber() << "("
                              << operandState.getScale() << "), ");
      scales.push_back(operandState.getScale());
    }
    if (scales.size() > 1) {
      if (scales[0] != scales[1]) {
        LLVM_DEBUG(llvm::dbgs() << "Different scales: " << scales[0] << ", "
                                << scales[1] << " for " << *op << "\n");
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return success();
  };

  auto getResultScales = [&](Operation* op, SmallVectorImpl<int64_t>& scales) {
    LLVM_DEBUG(llvm::dbgs() << "Result scales for " << op->getName() << ": ");
    SmallVector<OpResult> secretResults;
    this->getSecretResults(op, secretResults);

    for (auto result : secretResults) {
      auto resultState = this->getLatticeElement(result)->getValue();
      if (!resultState.isInitialized()) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "r" << cast<OpResult>(result).getResultNumber()
                              << "(" << resultState.getScale() << "), ");
      scales.push_back(resultState.getScale());
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  };

  LDBG() << "Backward analysis visiting: " << *op;
  llvm::TypeSwitch<Operation&>(*op)
      .template Case<arith::MulIOp, arith::MulFOp>([&](auto mulOp) {
        SmallVector<int64_t> resultScales;
        getResultScales(mulOp, resultScales);
        // there must be at least one secret result that has scale
        if (resultScales.empty()) {
          return;
        }
        SmallVector<int64_t> operandWithoutScaleIndices;
        SmallVector<int64_t> operandScales;
        if (failed(getOperandScales(mulOp, operandWithoutScaleIndices,
                                    operandScales))) {
          return;
        }
        // there must be at least one secret operand that has scale
        if (operandScales.empty()) {
          mulOp->emitError("No secret operand has scale");
          return;
        }
        // two operands have scale, succeed.
        if (operandScales.size() > 1) {
          return;
        }
        auto presentScale = operandScales[0];

        // propagate scale to other operand; this is guarded
        // by the loop for a weird reason: the secretness of the
        // non-scale-holding operand might not be initialized yet, depending
        // on the order in which the analyses run.
        for (auto otherIndex : operandWithoutScaleIndices) {
          auto scaleOther = ScaleModelT::evalMulScaleBackward(
              getLocalParam(mulOp.getResult()), resultScales[0], presentScale);
          propagate(mulOp->getOperand(otherIndex),
                    ScaleState<ScaleModelT>(scaleOther));
        }
      })
      .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        SmallVector<int64_t> resultScales;
        getResultScales(modReduceOp, resultScales);
        // there must be at least one secret result that has scale
        if (resultScales.empty()) {
          return;
        }
        SmallVector<int64_t> operandWithoutScaleIndices;
        SmallVector<int64_t> scales;
        if (failed(getOperandScales(modReduceOp, operandWithoutScaleIndices,
                                    scales))) {
          return;
        }
        // if all operands have scale, succeed.
        if (!scales.empty()) {
          return;
        }

        // propagate scale to operand
        auto resultScale = resultScales[0];
        // get level of the operand.
        auto newScale = ScaleModelT::evalModReduceScaleBackward(
            getLocalParam(modReduceOp.getInput()), resultScale);

        propagate(modReduceOp.getInput(), ScaleState<ScaleModelT>(newScale));
      })
      .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        // Do not back propagate through adjust scale op
        return;
      })
      .Default([&](auto& op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        this->getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<int64_t> scales;
        getResultScales(&op, scales);
        if (scales.empty()) {
          return;
        }

        // propagate the scale to all operands
        // including plaintext (non-secret)
        for (auto operand : op.getOperands()) {
          propagate(operand, ScaleState<ScaleModelT>(scales[0]));
        }
      });
  return success();
}

// instantiation
template class ScaleAnalysisBackward<BGVScaleModel>;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
int64_t getScale(Value value, DataFlowSolver* solver) {
  if (!isBlockLive(value.getParentBlock(), solver)) {
    return 0;
  }
  auto* lattice = solver->lookupState<ScaleLattice<ScaleModelT>>(value);
  if (!lattice) {
    assert(false && "ScaleLattice not found");
    return 0;
  }
  if (!lattice->getValue().isInitialized()) {
    assert(false && "ScaleLattice not initialized");
    return 0;
  }
  return lattice->getValue().getScale();
}

int64_t getScaleFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
    return 0;
  }
  return mgmtAttr.getScale();
}

template <typename ScaleModelT>
void annotateScale(Operation* top, DataFlowSolver* solver) {
  auto getIntegerAttr = [&](int scale) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), scale);
  };

  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      if (!isBlockLive(value.getParentBlock(), solver)) {
        return;
      }
      auto scale = getScale<ScaleModelT>(value, solver);
      LLVM_DEBUG(llvm::dbgs() << "Annotate scale " << scale
                              << " to value: " << value << "\n");
      setAttributeAssociatedWith(value, kArgScaleAttrName,
                                 getIntegerAttr(scale));
    }
  });
}

template int64_t getScale<BGVScaleModel>(Value, DataFlowSolver*);
template int64_t getScale<CKKSScaleModel>(Value, DataFlowSolver*);
template void annotateScale<BGVScaleModel>(Operation*, DataFlowSolver*);
template void annotateScale<CKKSScaleModel>(Operation*, DataFlowSolver*);

}  // namespace heir
}  // namespace mlir
