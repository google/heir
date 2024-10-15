#include "lib/Dialect/Polynomial/Transforms/PopulatePrimitiveRoot.h"

#include "lib/Dialect/Polynomial/Transforms/StaticRoots.h"
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_POLYPOPULATEPRIMITIVEROOT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

::mlir::polynomial::PrimitiveRootAttr populatePrimitiveRoot(Value poly) {
  auto polyType = dyn_cast<::mlir::polynomial::PolynomialType>(poly.getType());
  MLIRContext *context = polyType.getContext();

  auto ring = polyType.getRing();
  auto cmod = ring.getCoefficientModulus().getValue();
  auto polymod = ring.getPolynomialModulus().getPolynomial();

  unsigned rootBitWidth = (cmod - 1).getActiveBits();

  // TODO(#643): replace with a pass that computes roots as needed
  auto maybeRoot =
      rootBitWidth > 32
          ? roots::find64BitRoot(cmod, polymod.getDegree(), rootBitWidth)
          : roots::find32BitRoot(cmod, polymod.getDegree(), rootBitWidth);

  if (maybeRoot) {
    auto rootType = IntegerType::get(context, cmod.getBitWidth());
    auto rootAttr = IntegerAttr::get(
        rootType, (*maybeRoot).zextOrTrunc(cmod.getBitWidth()));
    auto degreeType = IntegerType::get(context, 32);
    auto degreeAttr = IntegerAttr::get(degreeType, polymod.getDegree() * 2);
    return ::mlir::polynomial::PrimitiveRootAttr::get(context, rootAttr,
                                                      degreeAttr);
  } else {
    return nullptr;
  }
}

namespace rewrites_populate {
// In an inner namespace to avoid conflicts with nttrewrite patterns
#include "lib/Dialect/Polynomial/Transforms/PopulatePrimitiveRoot.cpp.inc"
}  // namespace rewrites_populate

struct PolyPopulatePrimitiveRoot
    : impl::PolyPopulatePrimitiveRootBase<PolyPopulatePrimitiveRoot> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    rewrites_populate::populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
