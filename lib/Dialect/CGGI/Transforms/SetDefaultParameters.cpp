#include "include/Dialect/CGGI/Transforms/SetDefaultParameters.h"

#include "include/Dialect/CGGI/IR/CGGIAttributes.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_SETDEFAULTPARAMETERS
#include "include/Dialect/CGGI/Transforms/Passes.h.inc"

struct SetDefaultParameters
    : impl::SetDefaultParametersBase<SetDefaultParameters> {
  using SetDefaultParametersBase::SetDefaultParametersBase;

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext &context = getContext();
    unsigned defaultRlweDimension = 1;
    unsigned defaultPolyDegree = 1024;
    APInt defaultCmod = APInt::getOneBitSet(64, 32);
    IntegerAttr defaultCmodAttr =
        IntegerAttr::get(IntegerType::get(&context, 64), defaultCmod);

    // https://github.com/google/jaxite/blob/main/jaxite/jaxite_bool/bool_params.py
    unsigned defaultBskNoiseVariance = 65536;  // stdev = 2**8, var = 2**16
    unsigned defaultBskGadgetBaseLog = 4;
    unsigned defaultBskGadgetNumLevels = 6;
    unsigned defaultKskNoiseVariance = 268435456;  // stdev = 2**14, var = 2**28
    unsigned defaultKskGadgetBaseLog = 4;
    unsigned defaultKskGadgetNumLevels = 5;

    lwe::RLWEParamsAttr defaultRlweParams = lwe::RLWEParamsAttr::get(
        &context, defaultCmodAttr, defaultRlweDimension, defaultPolyDegree);
    CGGIParamsAttr defaultParams =
        CGGIParamsAttr::get(&context, defaultRlweParams,
                            defaultBskNoiseVariance, defaultBskGadgetBaseLog,
                            defaultBskGadgetNumLevels, defaultKskNoiseVariance,
                            defaultKskGadgetBaseLog, defaultKskGadgetNumLevels);

    auto walkResult = op->walk([&](Operation *op) {
      return llvm::TypeSwitch<Operation &, WalkResult>(*op)
          .Case<cggi::AndOp, cggi::OrOp, cggi::XorOp, cggi::NotOp, cggi::Lut2Op,
                cggi::Lut3Op>([&](auto op) {
            op.getOperation()->setAttr("cggi_params", defaultParams);
            return WalkResult::advance();
          })
          .Default([&](Operation &op) {
            if (llvm::isa<cggi::CGGIDialect>(op.getDialect())) {
              op.emitOpError() << "Found an unsupported cggi op";
              return WalkResult::interrupt();
            }
            // An unsupported op doesn't get any parameters set on it, and
            // that's OK.
            return WalkResult::advance();
          });
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
