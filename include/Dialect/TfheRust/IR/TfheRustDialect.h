#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_H_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_H_

#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project

// Generated headers (block clang-format from messing up order)
#include "include/Dialect/TfheRust/IR/TfheRustDialect.h.inc"

namespace mlir {
namespace heir {
namespace tfhe_rust {

template <typename ConcreteType>
class PassByReference
    : public TypeTrait::TraitBase<ConcreteType, PassByReference> {};

template <typename SksOp>
struct HoistConstantLikeOps : public OpRewritePattern<SksOp> {
  HoistConstantLikeOps(mlir::MLIRContext *context)
      : OpRewritePattern<SksOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(SksOp op,
                                PatternRewriter &rewriter) const override;
};

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_H_
