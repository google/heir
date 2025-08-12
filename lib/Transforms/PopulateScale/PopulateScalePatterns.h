#ifndef LIB_TRANSFORMS_POPULATESCALE_POPULATESCALEPATTERNS_H_
#define LIB_TRANSFORMS_POPULATESCALE_POPULATESCALEPATTERNS_H_

#include <cstdint>

#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

class AdjustScaleMaterializer {
 public:
  virtual int64_t deltaScale(int64_t scale, int64_t inputScale) const = 0;
};

template <typename MulOp>
struct ConvertAdjustScaleToMulPlain
    : public OpRewritePattern<mgmt::AdjustScaleOp> {
  using OpRewritePattern<mgmt::AdjustScaleOp>::OpRewritePattern;

  ConvertAdjustScaleToMulPlain(MLIRContext* context,
                               AdjustScaleMaterializer* materializer)
      : OpRewritePattern<mgmt::AdjustScaleOp>(context, /*benefit=*/1),
        materializer(materializer) {}

  LogicalResult matchAndRewrite(mgmt::AdjustScaleOp op,
                                PatternRewriter& rewriter) const override;

 private:
  AdjustScaleMaterializer* materializer;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_POPULATESCALE_POPULATESCALEPATTERNS_H_
