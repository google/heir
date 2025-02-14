#include "lib/Dialect/LWE/Transforms/SetDefaultParameters.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_SETDEFAULTPARAMETERS
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

void setLweParamsAttr(Value value, LWEParamsAttr attr) {
  LWECiphertextType type = mlir::cast<LWECiphertextType>(value.getType());
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
              if (mlir::isa<lwe::LWECiphertextType>(operand.get().getType())) {
                setLweParamsAttr(operand.get(), defaultLweParams);
              }
            }
            for (OpResult result : op.getResults()) {
              if (mlir::isa<lwe::LWECiphertextType>(result.getType())) {
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
              if (mlir::isa<lwe::LWECiphertextType>(ty)) {
                auto lweTy = mlir::cast<lwe::LWECiphertextType>(ty);
                newInputs.push_back(LWECiphertextType::get(
                    &context, lweTy.getEncoding(), defaultLweParams));
              } else {
                newInputs.push_back(ty);
              }
            }

            SmallVector<Type> newResults;
            for (Type ty : funcOp.getResultTypes()) {
              if (mlir::isa<lwe::LWECiphertextType>(ty)) {
                auto lweTy = mlir::cast<lwe::LWECiphertextType>(ty);
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
