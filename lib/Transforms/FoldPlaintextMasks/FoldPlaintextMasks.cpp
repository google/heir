#include "lib/Transforms/FoldPlaintextMasks/FoldPlaintextMasks.h"

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FOLDPLAINTEXTMASKS
#include "lib/Transforms/FoldPlaintextMasks/FoldPlaintextMasks.h.inc"

// If the value folds to an elements attr containing a 1-hot vector,
// return the one-hot vector, otherwise return failure.
FailureOr<SmallVector<APInt>> getIntegerOneHotVector(Value value) {
  auto ofr = getAsOpFoldResult(value);
  if (isa<Value>(ofr)) {
    return failure();
  }

  auto attr = cast<Attribute>(ofr);
  auto elementsAttr = dyn_cast<ElementsAttr>(attr);
  if (!elementsAttr) {
    return failure();
  }

  SmallVector<APInt> oneHotVector;
  for (auto element : elementsAttr.getValues<IntegerAttr>()) {
    APInt val = element.getValue();
    if (!val.isZero() && !val.isOne()) {
      return failure();
    }
    oneHotVector.push_back(val);
  }

  return oneHotVector;
}

FailureOr<SmallVector<APFloat>> getFloatOneHotVector(Value value) {
  auto ofr = getAsOpFoldResult(value);
  if (isa<Value>(ofr)) {
    return failure();
  }

  auto attr = cast<Attribute>(ofr);
  auto elementsAttr = dyn_cast<ElementsAttr>(attr);
  if (!elementsAttr) {
    return failure();
  }

  SmallVector<APFloat> oneHotVector;
  for (auto element : elementsAttr.getValues<FloatAttr>()) {
    APFloat val = element.getValue();
    if (val.isNonZero() && val != APFloat(val.getSemantics(), "1.0")) {
      return failure();
    }
    // clearSign maps -0 to +0
    val.clearSign();
    oneHotVector.push_back(val);
  }

  return oneHotVector;
}

bool isIntegerOneHotVector(Value value) {
  return succeeded(getIntegerOneHotVector(value));
}

bool isFloatOneHotVector(Value value) {
  return succeeded(getFloatOneHotVector(value));
}

DenseIntElementsAttr intersectIntMasks(Value mask1, Value mask2) {
  SmallVector<APInt> oneHot1 = getIntegerOneHotVector(mask1).value();
  SmallVector<APInt> oneHot2 = getIntegerOneHotVector(mask2).value();
  SmallVector<APInt> intersected;
  for (size_t i = 0; i < oneHot1.size(); ++i) {
    intersected.push_back(oneHot1[i] & oneHot2[i]);
  }
  return DenseIntElementsAttr::get(cast<RankedTensorType>(mask1.getType()),
                                   intersected);
}

DenseFPElementsAttr intersectFloatMasks(Value mask1, Value mask2) {
  SmallVector<APFloat> oneHot1 = getFloatOneHotVector(mask1).value();
  SmallVector<APFloat> oneHot2 = getFloatOneHotVector(mask2).value();
  SmallVector<APFloat> intersected;
  for (size_t i = 0; i < oneHot1.size(); ++i) {
    APFloat one(oneHot1[i].getSemantics(), "1.0");
    if (oneHot1[i] == one && oneHot2[i] == one) {
      intersected.push_back(oneHot1[i]);
      continue;
    }
    APFloat zeroVal(oneHot1[i].getSemantics(), "0.0");
    intersected.push_back(zeroVal);
  }
  return DenseFPElementsAttr::get(cast<RankedTensorType>(mask1.getType()),
                                  intersected);
}

DenseElementsAttr intersectMasks(Value mask1, Value mask2) {
  if (isa<IntegerType>(getElementTypeOrSelf(mask1.getType()))) {
    return intersectIntMasks(mask1, mask2);
  }

  return intersectFloatMasks(mask1, mask2);
}

bool isOneHotVector(Value mask) {
  if (isa<IntegerType>(getElementTypeOrSelf(mask.getType()))) {
    return isIntegerOneHotVector(mask);
  }

  return isFloatOneHotVector(mask);
}

namespace fold_plaintext_masks {
#include "lib/Transforms/FoldPlaintextMasks/Patterns.cpp.inc"
}  // namespace fold_plaintext_masks

struct FoldPlaintextMasks : impl::FoldPlaintextMasksBase<FoldPlaintextMasks> {
  using FoldPlaintextMasksBase::FoldPlaintextMasksBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    fold_plaintext_masks::populateWithGenerated(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
