#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTPATTERNS_H_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTPATTERNS_H_

#include "include/Dialect/TfheRust/IR/TfheRustOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// Moves tfhe_rust ops that only depend on constants and the server key as
// early as possible in the IR. Works generically across ops that have the
// tfhe_rust.server_key as the first operand.
template <typename SksOp>
struct HoistConstantLikeOps : public OpRewritePattern<SksOp> {
  HoistConstantLikeOps(mlir::MLIRContext *context)
      : OpRewritePattern<SksOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(SksOp op,
                                PatternRewriter &rewriter) const override;
};

struct HoistGenerateLookupTable : HoistConstantLikeOps<GenerateLookupTableOp> {
  HoistGenerateLookupTable(mlir::MLIRContext *context)
      : HoistConstantLikeOps(context, /*benefit=*/1) {}
};
struct HoistCreateTrivial : HoistConstantLikeOps<CreateTrivialOp> {};

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTPATTERNS_H_
