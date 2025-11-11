#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h"

#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "llvm/include/llvm/ADT/APFloat.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ACTIVATIONCANONICALIZATIONS
#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h.inc"

static bool IsOne(mlir::Attribute attr) {
  mlir::FloatType floatTy;
  llvm::APFloat floatVal(0.0);

  if (auto splattr = mlir::dyn_cast_or_null<mlir::SplatElementsAttr>(attr)) {
    floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(splattr.getElementType());
    floatVal = splattr.getValues<llvm::APFloat>()[0];
  } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr)) {
    floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(floatAttr.getType());
    floatVal = floatAttr.getValue();
  }

  if (!floatTy) return false;
  auto one =
      llvm::APFloat::getOne(floatTy.getFloatSemantics(), /*Negative=*/false);
  return floatVal == one;
}

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Transforms/ActivationCanonicalizations/Rewrites.cpp.inc"

struct ActivationCanonicalizations
    : impl::ActivationCanonicalizationsBase<ActivationCanonicalizations> {
  using ActivationCanonicalizationsBase::ActivationCanonicalizationsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
