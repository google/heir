#include "lib/Dialect/CGGI/Transforms/SetDefaultParameters.h"

#include <vector>

#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_SETDEFAULTPARAMETERS
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

struct SetDefaultParameters
    : impl::SetDefaultParametersBase<SetDefaultParameters> {
  using SetDefaultParametersBase::SetDefaultParametersBase;

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext &context = getContext();
    unsigned defaultRlweDimension = 1;
    APInt defaultCmod = APInt::getOneBitSet(64, 32);
    std::vector<::mlir::polynomial::IntMonomial> monomials;
    monomials.emplace_back(1, 1024);
    monomials.emplace_back(1, 0);
    ::mlir::polynomial::IntPolynomial defaultPolyIdeal =
        ::mlir::polynomial::IntPolynomial::fromMonomials(monomials).value();

    // https://github.com/google/jaxite/blob/main/jaxite/jaxite_bool/bool_params.py
    unsigned defaultBskNoiseVariance = 65536;  // stdev = 2**8, var = 2**16
    unsigned defaultBskGadgetBaseLog = 4;
    unsigned defaultBskGadgetNumLevels = 6;
    unsigned defaultKskNoiseVariance = 268435456;  // stdev = 2**14, var = 2**28
    unsigned defaultKskGadgetBaseLog = 4;
    unsigned defaultKskGadgetNumLevels = 5;
    auto intType = IntegerType::get(&context, 64);

    lwe::RLWEParamsAttr defaultRlweParams = lwe::RLWEParamsAttr::get(
        &context, defaultRlweDimension,
        ::mlir::polynomial::RingAttr::get(
            intType, IntegerAttr::get(intType, defaultCmod),
            polynomial::IntPolynomialAttr::get(&context, defaultPolyIdeal)));
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
