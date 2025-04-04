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

// RLWE scheme pattern to rewrite extract ops as a multiplication by a one-hot
// plaintext, followed by a rotate.
template <typename ExtractOp, typename MulPlainOp, typename RotateOp>
struct ConvertExtract : public OpRewritePattern<ExtractOp> {
  ConvertExtract(mlir::MLIRContext *context)
      : OpRewritePattern<ExtractOp>(context) {}

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter &rewriter) const override {
    // Not-directly-constant offsets could be supported by using -sccp or
    // including a constant propagation analysis in this pass. A truly
    // non-constant extract op seems unlikely, given that most programs should
    // be using rotate instead of extractions, and that we mainly have extract
    // as a terminating op for IRs that must output a secret<scalar> type.
    auto offsetOp = op.getOffset().template getDefiningOp<arith::ConstantOp>();
    if (!offsetOp) {
      return op.emitError()
             << "Expected extract offset arg to be constant integer, found "
             << op.getOffset();
    }
    auto offsetAttr = llvm::dyn_cast<IntegerAttr>(offsetOp.getValue());
    if (!offsetAttr) {
      return op.emitError()
             << "Expected extract offset arg to be constant integer, found "
             << op.getOffset();
    }
    int64_t offset = offsetAttr.getInt();

    auto ctTy = op.getInput().getType();
    auto ring = ctTy.getCiphertextSpace().getRing();
    auto degree = ring.getPolynomialModulus().getPolynomial().getDegree();
    auto elementTy =
        op.getOutput().getType().getApplicationData().getMessageType();
    if (!elementTy.isIntOrFloat()) {
      op.emitError()
          << "Expected extract op to extract scalar from tensor "
             "type, but found input underlying type "
          << op.getInput().getType().getApplicationData().getMessageType()
          << " and output underlying type "
          << op.getOutput().getType().getApplicationData().getMessageType();
    }
    auto tensorTy = RankedTensorType::get({degree}, elementTy);

    SmallVector<Attribute> oneHotCleartextAttrs;
    oneHotCleartextAttrs.reserve(degree);
    for (size_t i = 0; i < degree; ++i) {
      auto attrVal = (i == (unsigned int)offset) ? 1 : 0;
      if (elementTy.isIntOrIndex()) {
        oneHotCleartextAttrs.push_back(
            rewriter.getIntegerAttr(elementTy, attrVal));
      } else {
        oneHotCleartextAttrs.push_back(
            rewriter.getFloatAttr(elementTy, attrVal));
      }
    }

    auto b = ImplicitLocOpBuilder(rewriter.getUnknownLoc(), rewriter);
    auto oneHotCleartext =
        b.create<arith::ConstantOp>(
             tensorTy, DenseElementsAttr::get(tensorTy, oneHotCleartextAttrs))
            .getResult();
    auto plaintextTy = lwe::NewLWEPlaintextType::get(
        op.getContext(), ctTy.getApplicationData(), ctTy.getPlaintextSpace());
    auto oneHotPlaintext =
        b.create<lwe::RLWEEncodeOp>(plaintextTy, oneHotCleartext,
                                    ctTy.getPlaintextSpace().getEncoding(),
                                    ctTy.getPlaintextSpace().getRing())
            .getResult();
    auto plainMulOp = b.create<MulPlainOp>(op.getInput(), oneHotPlaintext);
    // inherit dialect attrs to all three new ops
    plainMulOp->setDialectAttrs(op->getDialectAttrs());
    auto plainMul = plainMulOp.getResult();
    auto rotated = b.create<RotateOp>(plainMul, offsetAttr);
    rotated->setDialectAttrs(op->getDialectAttrs());
    // It might make sense to move this op to the add-client-interface pass,
    // but it also seems like a backend implementation detail, and not part
    // of RLWE schemes generally.
    auto recast = b.create<lwe::ReinterpretApplicationDataOp>(
                       op.getOutput().getType(), rotated.getResult())
                      .getResult();
    rotated->setDialectAttrs(op->getDialectAttrs());
    rewriter.replaceOp(op, recast);
    return success();
  }
};

// A pattern to canonicalize scheme ops that are ciphertext-plaintext ops, but
// which allow the ciphertext to be in either operand. This is necessary
// because lower-level ops like openfhe may require the ciphertext to be in a
// specific operand.
template <typename Op>
struct PutCiphertextInFirstOperand : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    if (isa<lwe::NewLWEPlaintextType>(lhs.getType()) &&
        isa<lwe::NewLWECiphertextType>(rhs.getType())) {
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
