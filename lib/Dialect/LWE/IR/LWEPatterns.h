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
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

// RLWE scheme pattern to rewrite extract ops as a multiplication by a one-hot
// plaintext, followed by a rotate.
template <typename RlweExtractOp, typename RlweMulPlainOp,
          typename RlweRotateOp>
struct ConvertRlweExtractOp : public OpConversionPattern<RlweExtractOp> {
  ConvertRlweExtractOp(mlir::MLIRContext *context)
      : OpConversionPattern<RlweExtractOp>(context) {}

  using OpConversionPattern<RlweExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RlweExtractOp op, typename RlweExtractOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    // Not-directly-constant offsets could be supported by using -sccp or
    // including a constant propagation analysis in this pass. A truly
    // non-constant extract op seems unlikely, given that most programs should
    // be using rotate instead of extractions, and that we mainly have extract
    // as a terminating op for IRs that must output a secret<scalar> type.
    auto offsetOp =
        adaptor.getOffset().template getDefiningOp<arith::ConstantOp>();
    if (!offsetOp) {
      return op.emitError()
             << "Expected extract offset arg to be constant integer, found "
             << adaptor.getOffset();
    }
    auto offsetAttr = llvm::dyn_cast<IntegerAttr>(offsetOp.getValue());
    if (!offsetAttr) {
      return op.emitError()
             << "Expected extract offset arg to be constant integer, found "
             << adaptor.getOffset();
    }
    int64_t offset = offsetAttr.getInt();

    auto ctTy = op.getInput().getType();
    auto ring = ctTy.getRlweParams().getRing();
    auto degree = ring.getPolynomialModulus().getPolynomial().getDegree();
    auto elementTy =
        dyn_cast<IntegerType>(op.getOutput().getType().getUnderlyingType());
    if (!elementTy) {
      op.emitError() << "Expected extract op to extract scalar from tensor "
                        "type, but found input underlying type "
                     << op.getInput().getType().getUnderlyingType()
                     << " and output underlying type "
                     << op.getOutput().getType().getUnderlyingType();
    }
    auto tensorTy = RankedTensorType::get({degree}, elementTy);

    SmallVector<Attribute> oneHotCleartextAttrs;
    oneHotCleartextAttrs.reserve(degree);
    for (size_t i = 0; i < degree; ++i) {
      oneHotCleartextAttrs.push_back(rewriter.getIntegerAttr(
          elementTy, i == (unsigned int)offset ? 1 : 0));
    }

    auto b = ImplicitLocOpBuilder(rewriter.getUnknownLoc(), rewriter);
    auto oneHotCleartext =
        b.create<arith::ConstantOp>(
             tensorTy, DenseElementsAttr::get(tensorTy, oneHotCleartextAttrs))
            .getResult();
    auto plaintextTy = lwe::RLWEPlaintextType::get(
        op.getContext(), ctTy.getEncoding(), ring, tensorTy);
    auto oneHotPlaintext =
        b.create<lwe::RLWEEncodeOp>(plaintextTy, oneHotCleartext,
                                    ctTy.getEncoding(), ring)
            .getResult();
    auto plainMul =
        b.create<RlweMulPlainOp>(adaptor.getInput(), oneHotPlaintext)
            .getResult();
    auto rotated = b.create<RlweRotateOp>(plainMul, offsetAttr);
    // It might make sense to move this op to the add-client-interface pass,
    // but it also seems like an implementation detail of OpenFHE, and not part
    // of RLWE schemes generally.
    auto recast = b.create<lwe::ReinterpretUnderlyingTypeOp>(
                       op.getOutput().getType(), rotated.getResult())
                      .getResult();
    rewriter.replaceOp(op, recast);
    return success();
  }
};

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_IR_LWEPATTERNS_H_
