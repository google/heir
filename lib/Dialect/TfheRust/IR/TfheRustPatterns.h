#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTPATTERNS_H_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTPATTERNS_H_

#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// Moves tfhe_rust ops that only depend on constants and the server key as
// early as possible in the IR. Works generically across ops that have the
// tfhe_rust.server_key as the first operand.

struct HoistGenerateLookupTable
    : public OpRewritePattern<GenerateLookupTableOp> {
  HoistGenerateLookupTable(mlir::MLIRContext* context)
      : OpRewritePattern<GenerateLookupTableOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(GenerateLookupTableOp op,
                                PatternRewriter& rewriter) const override;
};

struct HoistCreateTrivial : public OpRewritePattern<CreateTrivialOp> {
  HoistCreateTrivial(mlir::MLIRContext* context)
      : OpRewritePattern<CreateTrivialOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(CreateTrivialOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTPATTERNS_H_
