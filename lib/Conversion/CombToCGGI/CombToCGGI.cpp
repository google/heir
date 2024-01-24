#include "include/Conversion/CombToCGGI/CombToCGGI.h"

#include <cassert>
#include <cstdint>

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/Comb/IR/CombOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWEOps.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Conversion/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::comb {

#define GEN_PASS_DEF_COMBTOCGGI
#include "include/Conversion/CombToCGGI/CombToCGGI.h.inc"

namespace {

// buildSelectTruthTable recursively creates arithmetic operations to compute a
// lookup table on plaintext integers.
Value buildSelectTruthTable(Location loc, OpBuilder &b, Value t, Value f,
                            const APInt &lut, ArrayRef<Value> lutInputs) {
  int tableSize = lut.getBitWidth();
  assert(tableSize == (1ull << lutInputs.size()));
  if (tableSize == 1) {
    return lut.isZero() ? f : t;
  }

  int halfSize = tableSize / 2;
  auto firstHalf = lut.extractBits(halfSize, 0);
  auto lastHalf = lut.extractBits(halfSize, halfSize);

  Value selectTrue =
      buildSelectTruthTable(loc, b, t, f, firstHalf, lutInputs.drop_back());
  Value selectFalse =
      buildSelectTruthTable(loc, b, t, f, lastHalf, lutInputs.drop_back());
  return b.create<arith::SelectOp>(loc, lutInputs.back(), selectTrue,
                                   selectFalse);
}

bool isCiphertextOrSecret(Type type) {
  if (isa<secret::SecretType>(type) || isa<lwe::LWECiphertextType>(type)) {
    return true;
  }
  if (ShapedType shapedType = dyn_cast<ShapedType>(type)) {
    return isCiphertextOrSecret(shapedType.getElementType());
  }
  return false;
}

// equivalentMultiBitAndMemRefchecks whether the candidateMultiBit integer type
// is equivalent to the candidateMemRef type.
// Return true if the candidateMemRef is a memref of single bits with
// size equal to the number of bits of the candidateMultiBit.
bool equivalentMultiBitAndMemRef(Type candidateMultiBit, Type candidateMemRef) {
  if (auto multiBitTy = dyn_cast<IntegerType>(candidateMultiBit)) {
    if (auto memrefTy = dyn_cast<MemRefType>(candidateMemRef)) {
      auto eltTy = dyn_cast<IntegerType>(memrefTy.getElementType());
      if (eltTy && multiBitTy.getWidth() ==
                       memrefTy.getNumElements() * eltTy.getWidth()) {
        return true;
      }
    }
  }
  return false;
}

LogicalResult convertWriteOpInterface(Operation *op, SmallVector<Value> indices,
                                      ConversionPatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  Type valueToStore = op->getOperand(0).getType();
  return llvm::TypeSwitch<Type, LogicalResult>(valueToStore)
      // Plaintext integer into memref
      .Case<IntegerType>([&](auto valType) {
        auto ctMemRefTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto ctTy = ctMemRefTy.getElementType();
        auto encoding = cast<lwe::LWECiphertextType>(ctTy).getEncoding();
        auto ptxtTy = lwe::LWEPlaintextType::get(b.getContext(), encoding);

        // Get i-th bit of input and insert the bit into the memref of
        // ciphertexts.
        auto loop = b.create<mlir::affine::AffineForOp>(0, valType.getWidth());
        b.setInsertionPointToStart(loop.getBody());
        auto idx = loop.getInductionVar();

        auto one = b.create<arith::ConstantOp>(
            valType, rewriter.getIntegerAttr(valType, 1));
        auto shiftAmount = b.create<arith::IndexCastOp>(valType, idx);
        auto bitMask = b.create<arith::ShLIOp>(valType, one, shiftAmount);
        auto andOp = b.create<arith::AndIOp>(op->getOperand(0), bitMask);
        auto shifted = b.create<arith::ShRSIOp>(andOp, shiftAmount);
        auto bitValue = b.create<arith::TruncIOp>(b.getI1Type(), shifted);
        auto ctValue = b.create<lwe::TrivialEncryptOp>(
            ctTy, b.create<lwe::EncodeOp>(ptxtTy, bitValue, encoding),
            lwe::LWEParamsAttr());

        indices.push_back(idx);
        b.create<memref::StoreOp>(ctValue, op->getOperand(1), indices);

        rewriter.eraseOp(op);
        return success();
      })
      .Case<MemRefType>([&](MemRefType valType) {
        auto &memRef = op->getOpOperand(1);
        MemRefType toMemRefTy = dyn_cast<MemRefType>(memRef.get().getType());
        int rank = toMemRefTy.getRank();

        // A store op with a memref value to store must have
        // originated from a secret encoding a multi-bit value. Under type
        // conversion, the op is storing a memref<BITSIZE!ct_ty> into a
        // memref of the form memref<?xBITSIZE!ct_ty>.

        // Subview the storage memref into a rank-reduced memref at the
        // storage indices.
        //   * Offset: offset at the storage index [indices, 0]
        //   * Strides: match original memref [1, 1, ..., 1]
        //   * Sizes: rank-reduce to the last dim [1, 1, ..., BITSIZE]
        SmallVector<OpFoldResult> offsets = getAsOpFoldResult(indices);
        offsets.push_back(OpFoldResult(b.getIndexAttr(0)));

        Attribute oneIdxAttr = rewriter.getIndexAttr(1);
        SmallVector<OpFoldResult> strides(rank, oneIdxAttr);
        SmallVector<OpFoldResult> sizes(rank - 1, oneIdxAttr);
        sizes.push_back(rewriter.getIndexAttr(toMemRefTy.getShape()[rank - 1]));

        mlir::Type memRefType =
            mlir::memref::SubViewOp::inferRankReducedResultType(
                valType.getShape(), toMemRefTy, offsets, sizes, strides);
        auto subview =
            b.create<memref::SubViewOp>(cast<MemRefType>(memRefType),
                                        memRef.get(), offsets, sizes, strides);
        b.create<memref::CopyOp>(op->getOpOperand(0).get(), subview);
        rewriter.eraseOp(op);
        return success();
      })
      .Case(
          [&](Type) {
            op->emitError()
                << "expected integer or memref to store in ciphertext memref";
            return failure();
          });
}

LogicalResult convertReadOpInterface(Operation *op, SmallVector<Value> indices,
                                     ConversionPatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  MemRefType fromMemRefTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
  MemRefType toMemRefTy = dyn_cast<MemRefType>(op->getResult(0).getType());
  int rank = fromMemRefTy.getRank();

  // A load op with a memref value to store must have
  // originated from a secret encoding a multi-bit value. Under type
  // conversion, the op is loading a memref<BITSIZE!ct_ty> from a
  // memref<?xBITSIZE!ct_ty>.

  // Subview the storage memref into a rank-reduced memref at the
  // storage indices.
  //   * Offset: offset at the storage index [indices, 0]
  //   * Strides: match original memref [1, 1, ..., 1]
  //   * Sizes: rank-reduce to the last dim [1, 1, ..., BITSIZE]
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets = getAsOpFoldResult(indices);
  offsets.push_back(OpFoldResult(b.getIndexAttr(0)));
  SmallVector<OpFoldResult> strides(rank, oneIdxAttr);
  SmallVector<OpFoldResult> sizes(rank - 1, oneIdxAttr);
  sizes.push_back(rewriter.getIndexAttr(fromMemRefTy.getShape()[rank - 1]));

  // We need to calculate the resulting subview shape which may have a dynamic
  // offset.
  mlir::Type memRefType = mlir::memref::SubViewOp::inferRankReducedResultType(
      toMemRefTy.getShape(), fromMemRefTy, offsets, sizes, strides);
  // If the offsets are dynamic and the resulting type does not match the
  // converted output type, we must allocate and copy into one with a static 0
  // offset. Otherwise we can return the subview.
  auto subViewOp = b.create<memref::SubViewOp>(
      cast<MemRefType>(memRefType), op->getOperand(0), offsets, sizes, strides);
  if (memRefType != toMemRefTy) {
    auto allocOp = b.create<memref::AllocOp>(toMemRefTy);
    b.create<memref::CopyOp>(subViewOp, allocOp);
    rewriter.replaceOp(op, allocOp);
  } else {
    rewriter.replaceOp(op, subViewOp);
  }
  return success();
}
}  // namespace

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx, int minBitWidth)
      : minBitWidth(minBitWidth) {
    addConversion([](Type type) { return type; });

    // Convert secret types to LWE ciphertext types
    addConversion([ctx, this](secret::SecretType type) -> Type {
      return getLWECiphertextForInt(ctx, type.getValueType());
    });
  }

  Type getLWECiphertextForInt(MLIRContext *ctx, Type type) const {
    if (IntegerType intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() == 1) {
        return lwe::LWECiphertextType::get(
            ctx, lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, minBitWidth),
            lwe::LWEParamsAttr());
      }
      return MemRefType::get(
          {intType.getWidth()},
          getLWECiphertextForInt(ctx, IntegerType::get(ctx, 1)));
    }
    ShapedType shapedType = dyn_cast<ShapedType>(type);
    assert(shapedType &&
           "expected shaped secret type for a non-integer secret");
    assert(isa<IntegerType>(shapedType.getElementType()) &&
           "expected integer element types for shaped secret types");
    auto elementType = getLWECiphertextForInt(ctx, shapedType.getElementType());
    SmallVector<int64_t> newShape = {shapedType.getShape().begin(),
                                     shapedType.getShape().end()};
    if (auto elementShape = dyn_cast<ShapedType>(elementType)) {
      // Flatten the element shape with the original shape
      newShape.insert(newShape.end(), elementShape.getShape().begin(),
                      elementShape.getShape().end());
      return MemRefType::get(newShape, elementShape.getElementType());
    }
    return shapedType.cloneWith(newShape, elementType);
  }

  int minBitWidth;
};

class SecretGenericOpTypeConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Block *originalBlock = op->getBlock();
    Block &opEntryBlock = op.getRegion().front();

    secret::YieldOp yieldOp =
        dyn_cast<secret::YieldOp>(op.getRegion().back().getTerminator());

    // Split the parent block of the generic op, so that we have a
    // clear insertion point for inlining.
    Block *newBlock = rewriter.splitBlock(originalBlock, Block::iterator(op));

    // mergeBlocks does not replace the original block values with the inputs to
    // secret.generic, so we manually replace them here. This lifts the internal
    // plaintext integer values within the secret.generic body to their original
    // secret values.
    auto genericInputs = adaptor.getInputs();
    for (unsigned i = 0; i < opEntryBlock.getNumArguments(); i++) {
      rewriter.replaceAllUsesWith(opEntryBlock.getArgument(i),
                                  genericInputs[i]);
    }

    // In addition to lifting the plaintext arguments, we also lift the output
    // types to secrets. This is required for any truth tables that have secret
    // inputs and any new data we allocate that will be yielded as a secret.
    // If this doesn't occur, then the type conversion framework is unable to
    // update the uses of the op results (due to type mismatches?) in any calls
    // to replaceOp or replaceAllUsesWith.
    rewriter.startOpModification(op);
    const SecretTypeConverter *secretConverter =
        static_cast<const SecretTypeConverter *>(typeConverter);
    opEntryBlock.walk<WalkOrder::PreOrder>([&](Operation *op) {
      bool allocatedSecretData = false;
      if (memref::AllocOp allocOp = dyn_cast<memref::AllocOp>(op)) {
        if (llvm::any_of(op->getUsers(), [&](Operation *op) {
              return isa<secret::YieldOp>(op);
            })) {
          allocatedSecretData = true;
        }
      }
      bool ciphertextArg =
          llvm::any_of(op->getOperands(), [&](const Value &val) {
            return isCiphertextOrSecret(val.getType());
          });
      if (ciphertextArg || allocatedSecretData) {
        for (unsigned i = 0; i < op->getNumResults(); i++) {
          op->getResult(i).setType(secretConverter->getLWECiphertextForInt(
              getContext(), op->getResult(i).getType()));
        }
      }
    });

    rewriter.finalizeOpModification(op);

    // Inline the secret.generic internal region, moving all of the operations
    // to the parent region.
    rewriter.inlineRegionBefore(op.getRegion(), newBlock);
    rewriter.replaceOp(op, yieldOp->getOperands());
    rewriter.mergeBlocks(&opEntryBlock, originalBlock, genericInputs);
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(yieldOp);
    return success();
  }
};

// ConvertTruthTableOp converts op arguments to trivially encoded LWE
// ciphertexts when at least one argument is an LWE ciphertext.
struct ConvertTruthTableOp : public OpConversionPattern<TruthTableOp> {
  ConvertTruthTableOp(mlir::MLIRContext *context)
      : OpConversionPattern<TruthTableOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TruthTableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3) {
      op->emitError() << "expected 3 truth table arguments to lower to CGGI";
    }

    MLIRContext *ctx = getContext();
    bool ciphertextArg = std::any_of(
        adaptor.getOperands().begin(), adaptor.getOperands().end(),
        [&](const Value &val) { return isCiphertextOrSecret(val.getType()); });

    SmallVector<mlir::Value, 4> lutInputs;
    const SecretTypeConverter *secretConverter =
        static_cast<const SecretTypeConverter *>(typeConverter);
    for (Value val : adaptor.getOperands()) {
      auto integerTy = dyn_cast<IntegerType>(val.getType());
      // If any of the arguments to the truth table are ciphertexts, we must
      // encode and trivially encrypt the plaintext integers arguments.
      if (ciphertextArg && integerTy) {
        assert(integerTy.getWidth() == 1 && "LUT inputs should be single-bit");
        auto ctxtTy = secretConverter->getLWECiphertextForInt(ctx, integerTy);
        auto encoding = cast<lwe::LWECiphertextType>(ctxtTy).getEncoding();
        auto ptxtTy = lwe::LWEPlaintextType::get(ctx, encoding);

        lutInputs.push_back(rewriter.create<lwe::TrivialEncryptOp>(
            op.getLoc(), ctxtTy,
            rewriter.create<lwe::EncodeOp>(op.getLoc(), ptxtTy, val, encoding),
            lwe::LWEParamsAttr()));
      } else {
        lutInputs.push_back(val);
      }
    }

    if (!ciphertextArg) {
      // When all three arguments are plaintext, create a truth table op out of
      // arithmetic statements.
      Value t = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value f = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.replaceOp(
          op, buildSelectTruthTable(op.getLoc(), rewriter, t, f,
                                    op.getLookupTable().getValue(), lutInputs));
    } else {
      rewriter.replaceOp(op, rewriter.create<cggi::Lut3Op>(
                                 op.getLoc(), lutInputs[0], lutInputs[1],
                                 lutInputs[2], op.getLookupTable()));
    }

    return success();
  }
};

// ConvertSecretCastOp removes secret.cast operations between multi-bit secret
// integers and tensors of single-bit secrets.
struct ConvertSecretCastOp : public OpConversionPattern<secret::CastOp> {
  ConvertSecretCastOp(mlir::MLIRContext *context)
      : OpConversionPattern<secret::CastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // If this is a cast from secret<i8> to secret<tensor<8xi1>> or vice
    // versa, replace with the cast's input.
    auto inputTy =
        cast<secret::SecretType>(op.getInput().getType()).getValueType();
    auto outputTy =
        cast<secret::SecretType>(op.getOutput().getType()).getValueType();

    if (equivalentMultiBitAndMemRef(inputTy, outputTy) ||
        equivalentMultiBitAndMemRef(outputTy, inputTy)) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }
    return failure();
  }
};

struct ConvertMemRefStoreOp : public OpConversionPattern<memref::StoreOp> {
  ConvertMemRefStoreOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::StoreOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return convertWriteOpInterface(
        op, {op.getIndices().begin(), op.getIndices().end()}, rewriter);
  }
};

struct ConvertAffineStoreOp
    : public OpConversionPattern<affine::AffineStoreOp> {
  ConvertAffineStoreOp(mlir::MLIRContext *context)
      : OpConversionPattern<affine::AffineStoreOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      affine::AffineStoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto indices = affine::expandAffineMap(rewriter, op.getLoc(),
                                           op.getAffineMap(), op.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    return convertWriteOpInterface(
        op, {indices.value().begin(), indices.value().end()}, rewriter);
  }
};

struct ConvertAffineLoadOp : public OpConversionPattern<affine::AffineLoadOp> {
  ConvertAffineLoadOp(mlir::MLIRContext *context)
      : OpConversionPattern<affine::AffineLoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      affine::AffineLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto indices = affine::expandAffineMap(rewriter, op.getLoc(),
                                           op.getAffineMap(), op.getIndices());
    if (!indices) {
      op.emitError() << "expected affine access indices";
    }
    return convertReadOpInterface(
        op, {indices.value().begin(), indices.value().end()}, rewriter);
  }
};

struct ConvertMemRefLoadOp : public OpConversionPattern<memref::LoadOp> {
  ConvertMemRefLoadOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return convertReadOpInterface(op, op.getIndices(), rewriter);
  }
};

struct CombToCGGI : public impl::CombToCGGIBase<CombToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // TODO(#250): The lutSize here is fixed on the assumption that the comb
    // dialect is using ternary LUTs. Generalize lutSize by doing an analysis
    // pass on the input combinational operations and integers.
    int lutSize = 3;
    SecretTypeConverter typeConverter(context, lutSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    patterns
        .add<ConvertTruthTableOp, ConvertSecretCastOp,
             SecretGenericOpTypeConversion, ConvertMemRefStoreOp,
             ConvertAffineStoreOp, ConvertMemRefLoadOp, ConvertAffineLoadOp>(
            typeConverter, context);
    target.addIllegalOp<TruthTableOp, secret::CastOp, secret::GenericOp>();
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      // Legal only when the memref element type matches the stored type.
      return op.getMemRefType().getElementType() ==
             op.getValueToStore().getType();
    });
    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          // Legal only when the memref element type matches the stored type.
          return op.getMemRefType().getElementType() ==
                 op.getValueToStore().getType();
        });
    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      // Legal only when the memref element type matches the loaded type.
      return op.getMemRefType().getElementType() == op.getResult().getType();
    });
    target.addDynamicallyLegalOp<affine::AffineLoadOp>(
        [&](affine::AffineLoadOp op) {
          // Legal only when the memref element type matches the loaded type.
          return op.getMemRefType().getElementType() ==
                 op.getResult().getType();
        });
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::comb
