#include "lib/Transforms/ValidateScale/ValidateScale.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_VALIDATESCALE
#include "lib/Transforms/ValidateScale/ValidateScale.h.inc"

namespace {

std::optional<int64_t> getScale(Value val) {
  if (auto attr = mgmt::findMgmtAttrAssociatedWith(val)) {
    if (attr.getScale() != -1) {
      return attr.getScale();
    }
  }
  return std::nullopt;
}

std::optional<int> getLevel(Value val) {
  if (auto attr = mgmt::findMgmtAttrAssociatedWith(val)) {
    return attr.getLevel();
  }
  return std::nullopt;
}

LogicalResult operandAndResultScalesEqual(Operation* op) {
  SmallVector<int64_t> scales;
  for (auto operand : op->getOperands()) {
    if (auto scale = getScale(operand)) {
      scales.push_back(*scale);
    }
  }

  for (auto res : op->getResults()) {
    if (auto scale = getScale(res)) {
      scales.push_back(*scale);
    }
  }

  if (!llvm::all_equal(scales)) {
    return failure();
  }
  return success();
}

LogicalResult validateSecretValuesHaveScale(Operation* op,
                                            DataFlowSolver& solver) {
  LogicalResult res = success();
  DenseSet<Value> visited;
  walkValues(op, [&](Value value) {
    if (failed(res)) return;
    if (visited.insert(value).second) {
      if (!isa<SecretTypeInterface>(getElementTypeOrSelf(value.getType()))) {
        return;
      }
      bool secret = isSecret(value, &solver);
      auto scale = getScale(value);
      if (secret && !scale) {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          emitError(blockArg.getLoc(), "secret block argument has no scale");
        } else {
          value.getDefiningOp()->emitOpError(
              "secret result value has no scale");
        }
        res = failure();
      }
    }
  });
  return res;
}

struct ValidateScale : impl::ValidateScaleBase<ValidateScale> {
  using ValidateScaleBase::ValidateScaleBase;

  LogicalResult runCKKSValidation(const ckks::SchemeParam& param) {
    LogicalResult result = success();
    getOperation()->walk([&](Operation* op) {
      // 1. Additive & Container Operations
      if (isa<arith::AddFOp, arith::SubFOp, arith::AddIOp, arith::SubIOp,
              tensor::InsertSliceOp, tensor::InsertOp, ckks::AddOp, ckks::SubOp,
              ckks::AddPlainOp, ckks::SubPlainOp>(op)) {
        if (failed(operandAndResultScalesEqual(op))) {
          result = op->emitOpError(
              "operands and results must have all the same scale");
          return;
        }
      }

      // 2. Multiplication
      if (isa<arith::MulFOp, arith::MulIOp, ckks::MulOp, ckks::MulPlainOp>(
              op)) {
        auto lhsScale = getScale(op->getOperand(0));
        auto rhsScale = getScale(op->getOperand(1));
        auto resScale = getScale(op->getResult(0));
        if (lhsScale && rhsScale && resScale) {
          if (*resScale != *lhsScale + *rhsScale) {
            result = op->emitOpError(
                "result scale must equal the sum of operand scales");
            return;
          }
        }
      }

      // 3. ModReduce / Rescale
      if (isa<mgmt::ModReduceOp, ckks::RescaleOp>(op)) {
        Value input = op->getOperand(0);
        Value res = op->getResult(0);
        auto inScale = getScale(input);
        auto resScale = getScale(res);
        auto inLevel = getLevel(input);
        if (inScale && resScale && inLevel) {
          const auto& logqi = param.getLogqi();
          int64_t logqi_level = param.getLogDefaultScale();
          if (*inLevel >= 0 && *inLevel < static_cast<int>(logqi.size())) {
            logqi_level = static_cast<int64_t>(std::llround(logqi[*inLevel]));
          }
          if (*resScale != *inScale - logqi_level) {
            result = op->emitOpError(
                "result scale must equal input scale minus log2(q_i)");
            return;
          }
        }
      }

      // 4. AdjustScale
      if (auto adjustOp = dyn_cast<mgmt::AdjustScaleOp>(op)) {
        Value input = adjustOp.getInput();
        Value res = adjustOp.getResult();
        auto inScale = getScale(input);
        auto resScale = getScale(res);
        if (inScale && resScale) {
          if (*resScale < *inScale) {
            result = adjustOp.emitOpError(
                "target scale must be greater than or equal to input scale");
            return;
          }
        }
      }
    });
    return result;
  }

  LogicalResult runBGVValidation(const bgv::SchemeParam& param) {
    LogicalResult result = success();

    getOperation()->walk([&](Operation* op) {
      if (isa<arith::AddIOp, arith::AddFOp, bgv::AddOp, bgv::AddPlainOp,
              tensor::InsertSliceOp, tensor::InsertOp>(op)) {
        if (failed(operandAndResultScalesEqual(op))) {
          op->emitOpError("operands and results must have all the same scale");
          result = failure();
          return;
        }
      }
    });

    return result;
  }

  void runOnOperation() override {
    auto ckksSchemeParamAttr =
        getOperation()->getAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    auto bgvSchemeParamAttr =
        getOperation()->getAttr(bgv::BGVDialect::kSchemeParamAttrName);
    if (!ckksSchemeParamAttr && !bgvSchemeParamAttr) {
      return;
    }

    DataFlowSolver solver;
    mlir::dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitError("Failed to run SecretnessAnalysis.");
      signalPassFailure();
      return;
    }

    if (failed(validateSecretValuesHaveScale(getOperation(), solver))) {
      signalPassFailure();
      return;
    }

    if (ckksSchemeParamAttr) {
      auto ckksAttr =
          mlir::dyn_cast<ckks::SchemeParamAttr>(ckksSchemeParamAttr);
      if (ckksAttr) {
        auto param = ckks::getSchemeParamFromAttr(ckksAttr);
        if (failed(runCKKSValidation(param))) {
          signalPassFailure();
        }
      }
    }

    if (bgvSchemeParamAttr) {
      auto bgvAttr = mlir::dyn_cast<bgv::SchemeParamAttr>(bgvSchemeParamAttr);
      if (bgvAttr) {
        auto param = bgv::SchemeParam::getSchemeParamFromAttr(bgvAttr);
        if (failed(runBGVValidation(param))) {
          signalPassFailure();
        }
      }
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
