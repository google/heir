#ifndef LIB_DIALECT_LWE_IR_LWEPATTERNS_H_
#define LIB_DIALECT_LWE_IR_LWEPATTERNS_H_

#include <cstddef>
#include <cstdint>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

// A pattern to canonicalize scheme ops that are ciphertext-plaintext ops, but
// which allow the ciphertext to be in either operand. This is necessary
// because lower-level ops like openfhe may require the ciphertext to be in a
// specific operand.
template <typename Op>
struct PutCiphertextInFirstOperand : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const final {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    if (isa<lwe::LWEPlaintextType>(lhs.getType()) &&
        isa<lwe::LWECiphertextType>(rhs.getType())) {
      rewriter.modifyOpInPlace(op, [&] {
        op->setOperand(0, rhs);
        op->setOperand(1, lhs);
      });
      return success();
    }
    return failure();
  }
};

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_IR_LWEPATTERNS_H_
