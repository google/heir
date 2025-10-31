#ifndef LIB_TRANSFORMS_FORWARDINSERTSLICETOEXTRACTSLICE_FORWARDINSERTSLICETOEXTRACTSLICE_H_
#define LIB_TRANSFORMS_FORWARDINSERTSLICETOEXTRACTSLICE_FORWARDINSERTSLICETOEXTRACTSLICE_H_

#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/ForwardInsertSliceToExtractSlice/ForwardInsertSliceToExtractSlice.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/ForwardInsertSliceToExtractSlice/ForwardInsertSliceToExtractSlice.h.inc"

struct ForwardSingleInsertSliceToExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  ForwardSingleInsertSliceToExtractSlice(mlir::MLIRContext* context)
      : OpRewritePattern<tensor::ExtractSliceOp>(context, 3) {}

 public:
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override;

 private:
  FailureOr<OpFoldResult> getValueAtSlice(
      tensor::ExtractSliceOp originalExtractOp,
      TypedValue<RankedTensorType> tensor, PatternRewriter& rewriter) const;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_FORWARDINSERTSLICETOEXTRACTSLICE_FORWARDINSERTSLICETOEXTRACTSLICE_H_
