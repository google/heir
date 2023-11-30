#include "include/Dialect/LWE/Transforms/SetDefaultParameters.h"

#include "include/Dialect/LWE/IR/LWEOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_SETDEFAULTPARAMETERS
#include "include/Dialect/LWE/Transforms/Passes.h.inc"

void setLweParamsAttr(Value value, LWEParamsAttr attr) {
  LWECiphertextType type = value.getType().cast<LWECiphertextType>();
  // Calling setType is not recommended, but this pass is simple enough to do
  // it.
  value.setType(
      LWECiphertextType::get(value.getContext(), type.getEncoding(), attr));
}

struct SetDefaultParameters
    : impl::SetDefaultParametersBase<SetDefaultParameters> {
  using SetDefaultParametersBase::SetDefaultParametersBase;

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext &context = getContext();
    unsigned defaultLweDimension = 800;
    APInt defaultCmod = APInt::getOneBitSet(64, 32);
    IntegerAttr defaultCmodAttr =
        IntegerAttr::get(IntegerType::get(&context, 64), defaultCmod);

    lwe::LWEParamsAttr defaultLweParams =
        lwe::LWEParamsAttr::get(&context, defaultCmodAttr, defaultLweDimension);

    auto walkResult = op->walk([&](Operation *op) {
      return llvm::TypeSwitch<Operation &, WalkResult>(*op)
          .Case<lwe::TrivialEncryptOp>([&](auto op) {
            op.getOperation()->setAttr("params", defaultLweParams);
            return WalkResult::advance();
          })
          .Default([&](Operation &op) {
            for (OpOperand &operand : op.getOpOperands()) {
              if (operand.get().getType().isa<lwe::LWECiphertextType>()) {
                setLweParamsAttr(operand.get(), defaultLweParams);
              }
            }
            for (OpResult result : op.getResults()) {
              if (result.getType().isa<lwe::LWECiphertextType>()) {
                setLweParamsAttr(result, defaultLweParams);
              }
            }

            // Func-like ops require special handling because their signature
            // is technically stored as an attribute
            auto funcOp = dyn_cast<FunctionOpInterface>(op);
            if (!funcOp) {
              return WalkResult::advance();
            }

            SmallVector<Type> newInputs;
            for (Type ty : funcOp.getArgumentTypes()) {
              if (ty.isa<lwe::LWECiphertextType>()) {
                auto lweTy = ty.cast<lwe::LWECiphertextType>();
                newInputs.push_back(LWECiphertextType::get(
                    &context, lweTy.getEncoding(), defaultLweParams));
              } else {
                newInputs.push_back(ty);
              }
            }

            SmallVector<Type> newResults;
            for (Type ty : funcOp.getResultTypes()) {
              if (ty.isa<lwe::LWECiphertextType>()) {
                auto lweTy = ty.cast<lwe::LWECiphertextType>();
                newResults.push_back(LWECiphertextType::get(
                    &context, lweTy.getEncoding(), defaultLweParams));
              } else {
                newResults.push_back(ty);
              }
            }

            auto newFuncTy = FunctionType::get(&context, newInputs, newResults);
            funcOp.setType(newFuncTy);
            return WalkResult::advance();
          });
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
