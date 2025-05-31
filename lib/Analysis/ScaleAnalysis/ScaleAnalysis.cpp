#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"

#include <cassert>
#include <cstdint>
#include <functional>

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

//===----------------------------------------------------------------------===//
// ScaleModel
//===----------------------------------------------------------------------===//

int64_t BGVScaleModel::evalMulScale(const bgv::LocalParam &param, int64_t lhs,
                                    int64_t rhs) {
  const auto *schemeParam = param.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  return lhs * rhs % t;
}

int64_t BGVScaleModel::evalMulScaleBackward(const bgv::LocalParam &param,
                                            int64_t result, int64_t lhs) {
  const auto *schemeParam = param.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto lhsInv = multiplicativeInverse(APInt(64, lhs), APInt(64, t));
  return result * lhsInv.getSExtValue() % t;
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

int64_t BGVScaleModel::evalModReduceScaleBackward(
    const bgv::LocalParam &inputParam, int64_t resultScale) {
  const auto *schemeParam = inputParam.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto qi = schemeParam->getQi();
  auto level = inputParam.getCurrentLevel();
  return resultScale * (qi[level] % t) % t;
}

int64_t CKKSScaleModel::evalMulScale(const ckks::LocalParam &param, int64_t lhs,
                                     int64_t rhs) {
  // TODO(#1640): support high-precision scale management
  return lhs + rhs;
}

int64_t CKKSScaleModel::evalMulScaleBackward(const ckks::LocalParam &param,
                                             int64_t result, int64_t lhs) {
  // TODO(#1640): support high-precision scale management
  return result - lhs;
}

int64_t CKKSScaleModel::evalModReduceScale(const ckks::LocalParam &inputParam,
                                           int64_t scale) {
  const auto *schemeParam = inputParam.getSchemeParam();
  // TODO(#1640): rescale using logqi instead of logDefaultScale
  // auto logqi = schemeParam->getLogqi();
  // auto level = inputParam.getCurrentLevel();
  auto logDefaultScale = schemeParam->getLogDefaultScale();
  return scale - logDefaultScale;
}

int64_t CKKSScaleModel::evalModReduceScaleBackward(
    const ckks::LocalParam &inputParam, int64_t resultScale) {
  const auto *schemeParam = inputParam.getSchemeParam();
  // TODO(#1640): rescale using logqi instead of logDefaultScale
  // auto logqi = schemeParam->getLogqi();
  // auto level = inputParam.getCurrentLevel();
  auto logDefaultScale = schemeParam->getLogDefaultScale();
  return resultScale + logDefaultScale;
}

int64_t PlaintextScaleModel::evalMulScale(
    const PlaintextScaleModel::LocalParam &param, int64_t lhs, int64_t rhs) {
  return lhs + rhs;
}

int64_t PlaintextScaleModel::evalMulScaleBackward(
    const PlaintextScaleModel::LocalParam &param, int64_t result, int64_t lhs) {
  return result - lhs;
}

int64_t PlaintextScaleModel::evalModReduceScale(
    const PlaintextScaleModel::LocalParam &inputParam, int64_t scale) {
  return scale - inputParam.getDefaultLogScale();
}

int64_t PlaintextScaleModel::evalModReduceScaleBackward(
    const PlaintextScaleModel::LocalParam &inputParam, int64_t resultScale) {
  return resultScale + inputParam.getDefaultLogScale();
}
//===----------------------------------------------------------------------===//
// ScaleAnalysis (Forward)
//===----------------------------------------------------------------------===//

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
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Propagate " << state << " to " << value << "\n");
    }
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
        // adjust scale op is opaque, just do not propagate
        return;
      })
      .template Case<mgmt::InitOp>([&](auto initOp) {
        auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp.getResult());
        // if there is scale annotation, use it
        if (mgmtAttr && mgmtAttr.getScale() != 0) {
          propagate(initOp.getResult(), ScaleState(mgmtAttr.getScale()));
        }
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
template class ScaleAnalysis<PlaintextScaleModel>;

//===----------------------------------------------------------------------===//
// ScaleAnalysis (Backward)
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
LogicalResult ScaleAnalysisBackward<ScaleModelT>::visitOperation(
    Operation *op, ArrayRef<ScaleLattice *> operands,
    ArrayRef<const ScaleLattice *> results) {
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value);
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, const ScaleState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Back Propagate " << state << " to " << value << "\n");
    }
    propagateIfChanged(lattice, changed);
  };

  auto getSecretOrInittedOperands =
      [&](Operation *op, SmallVectorImpl<OpOperand *> &secretOperands) {
        this->getSecretOperands(op, secretOperands);
        for (auto &opOperand : op->getOpOperands()) {
          if (!this->isSecretInternal(op, opOperand.get()) &&
              isa_and_nonnull<mgmt::InitOp>(opOperand.get().getDefiningOp())) {
            // Treat it as if it were secret for the purpose of scale
            // propagation
            secretOperands.push_back(&opOperand);
          }
        }
      };

  auto getOperandScales =
      [&](Operation *op, SmallVectorImpl<int64_t> &operandWithoutScaleIndices,
          SmallVectorImpl<int64_t> &scales) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Operand scales for " << op->getName() << ": ");
        SmallVector<OpOperand *> secretOperands;
        getSecretOrInittedOperands(op, secretOperands);

        for (auto *operand : secretOperands) {
          auto operandState = getLatticeElement(operand->get())->getValue();
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
      };

  auto getResultScales = [&](Operation *op, SmallVectorImpl<int64_t> &scales) {
    LLVM_DEBUG(llvm::dbgs() << "Result scales for " << op->getName() << ": ");
    SmallVector<OpResult> secretResults;
    this->getSecretResults(op, secretResults);

    for (auto result : secretResults) {
      auto resultState = getLatticeElement(result)->getValue();
      if (!resultState.isInitialized()) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "r" << cast<OpResult>(result).getResultNumber()
                              << "(" << resultState.getScale() << "), ");
      scales.push_back(resultState.getScale());
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  };

  LLVM_DEBUG(llvm::dbgs() << "Backward analysis visiting: " << op->getName()
                          << "\n");
  llvm::TypeSwitch<Operation &>(*op)
      .template Case<arith::MulIOp, arith::MulFOp>([&](auto mulOp) {
        SmallVector<int64_t> resultScales;
        getResultScales(mulOp, resultScales);
        // there must be at least one secret result that has scale
        if (resultScales.empty()) {
          return;
        }
        SmallVector<int64_t> operandWithoutScaleIndices;
        SmallVector<int64_t> operandScales;
        getOperandScales(mulOp, operandWithoutScaleIndices, operandScales);
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

        // propagate scale to other operand
        auto scaleOther = ScaleModelT::evalMulScaleBackward(
            getLocalParam(mulOp.getResult()), resultScales[0], presentScale);
        propagate(mulOp->getOperand(operandWithoutScaleIndices[0]),
                  ScaleState(scaleOther));
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
        getOperandScales(modReduceOp, operandWithoutScaleIndices, scales);
        // if all operands have scale, succeed.
        if (!scales.empty()) {
          return;
        }

        // propagate scale to operand
        auto resultScale = resultScales[0];
        // get level of the operand.
        auto newScale = ScaleModelT::evalModReduceScaleBackward(
            getLocalParam(modReduceOp.getInput()), resultScale);

        propagate(modReduceOp.getInput(), ScaleState(newScale));
      })
      .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        // Do not back propagate through adjust scale op
        return;
      })
      .Default([&](auto &op) {
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
          propagate(operand, ScaleState(scales[0]));
        }
      });
  return success();
}

// instantiation
template class ScaleAnalysisBackward<BGVScaleModel>;
template class ScaleAnalysisBackward<CKKSScaleModel>;
template class ScaleAnalysisBackward<PlaintextScaleModel>;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

int64_t getScale(Value value, DataFlowSolver *solver) {
  auto *lattice = solver->lookupState<ScaleLattice>(value);
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

void annotateScale(Operation *top, DataFlowSolver *solver) {
  auto getIntegerAttr = [&](int scale) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), scale);
  };

  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      setAttributeAssociatedWith(value, kArgScaleAttrName,
                                 getIntegerAttr(getScale(value, solver)));
    }
  });
}

}  // namespace heir
}  // namespace mlir
