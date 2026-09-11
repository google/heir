#include "lib/Transforms/StampApproximationDomains/StampApproximationDomains.h"

#include "llvm/include/llvm/ADT/StringRef.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_STAMPAPPROXIMATIONDOMAINS
#include "lib/Transforms/StampApproximationDomains/StampApproximationDomains.h.inc"

namespace {

constexpr StringLiteral kDomainLower = "domain_lower";
constexpr StringLiteral kDomainUpper = "domain_upper";
constexpr StringLiteral kDegree = "degree";

struct StampApproximationDomains
    : public impl::StampApproximationDomainsBase<StampApproximationDomains> {
  using StampApproximationDomainsBase::StampApproximationDomainsBase;

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    getOperation()->walk([&](Operation* op) {
      if (op->hasAttr(kDomainLower)) return;
      if (isa<math::RsqrtOp>(op)) {
        op->setAttr(kDegree, builder.getI32IntegerAttr(rsqrtDegree));
        op->setAttr(kDomainLower, builder.getF64FloatAttr(rsqrtDomainLower));
        op->setAttr(kDomainUpper, builder.getF64FloatAttr(rsqrtDomainUpper));
      } else if (isa<math::ErfOp>(op)) {
        op->setAttr(kDegree, builder.getI32IntegerAttr(erfDegree));
        op->setAttr(kDomainLower, builder.getF64FloatAttr(-erfDomainBound));
        op->setAttr(kDomainUpper, builder.getF64FloatAttr(erfDomainBound));
      }
    });
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
