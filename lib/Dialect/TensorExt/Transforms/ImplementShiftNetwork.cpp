#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

#define DEBUG_TYPE "implement-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

// Create a tensor with zeros everywhere except for the indices specified in
// the input `indices` vector.
Value createMask(TypedValue<RankedTensorType> tensor,
                 const SmallVector<int64_t> &indices, IRRewriter &rewriter) {
  auto elementType = tensor.getType().getElementType();
  SmallVector<Attribute> maskAttrs(tensor.getType().getDimSize(0),
                                   rewriter.getIntegerAttr(elementType, 0));
  for (int64_t index : indices) {
    maskAttrs[index] = rewriter.getIntegerAttr(elementType, 1);
  }

  auto denseAttr = DenseElementsAttr::get(tensor.getType(), maskAttrs);
  auto constant =
      rewriter.create<arith::ConstantOp>(tensor.getLoc(), denseAttr);
  return constant.getResult();
}

Value rotateGroup(TypedValue<RankedTensorType> tensor,
                  const RotationGroup &group, int64_t ciphertextSize,
                  const Permutation &permutation, IRRewriter &rewriter) {
  std::optional<Value> result = std::nullopt;

  // Re-run the shift strategy on a single rotation group, and use the
  // rotatedIndices in each round to construct a mask and a rotation op.
  ShiftStrategy strategy;
  strategy.evaluate(permutation, group);

  [[maybe_unused]] int roundNum = 0;
  for (const ShiftRound &round : strategy.getRounds()) {
    int64_t rotationAmount = round.rotationAmount;
    if (round.rotatedIndices.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping shift by " << rotationAmount
                              << " because no inputs have that bit set\n");
      continue;
    }

    Value mask = createMask(tensor, round.rotatedIndices, rewriter);
    arith::MulIOp maskOp =
        rewriter.create<arith::MulIOp>(tensor.getLoc(), tensor, mask);
    Value rotated = rewriter.create<tensor_ext::RotateOp>(
        tensor.getLoc(), maskOp.getResult(),
        rewriter.create<arith::ConstantIntOp>(
            tensor.getLoc(), rewriter.getI32Type(), rotationAmount));

    if (result.has_value()) {
      result = rewriter.create<arith::AddIOp>(tensor.getLoc(), result.value(),
                                              rotated);
    } else {
      result = rotated;
    }
  }

  return result.has_value() ? result.value() : tensor;
}

LogicalResult convertPermuteOp(PermuteOp op,
                               VosVosErkinShiftNetworks &shiftNetworks,
                               int64_t ciphertextSize) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  IRRewriter rewriter(op.getContext());
  RankedTensorType tensorTy = op.getInput().getType();

  // Only support a 1-D tensor until sharding is supported
  if (op.getInput().getType().getRank() != 1) {
    return op.emitError("requires a one-dimensional tensor");
  }

  SmallVector<int64_t> permutation;

  // Convert the affine map to an explicit permutation, or else use the dense
  // array attr as an explicit permutation.
  auto affineMapAttr = dyn_cast<AffineMapAttr>(op.getPermutation());
  if (affineMapAttr) {
    AffineMap permutationMap = simplifyAffineMap(affineMapAttr.getValue());
    LLVM_DEBUG(llvm::dbgs()
               << "Expanding permutation from " << permutationMap << "\n");
    if (failed(makeExplicit1DMapping(permutationMap, tensorTy.getNumElements(),
                                     permutation)))
      return failure();
  } else {
    auto denseElementsAttr =
        dyn_cast<DenseIntElementsAttr>(op.getPermutation());
    if (denseElementsAttr) {
      permutation = llvm::map_to_vector(
          denseElementsAttr, [](const APInt &i) { return i.getSExtValue(); });
    } else {
      return failure();
    }
  }

  if (!isPermutation(permutation)) {
    auto diag = op.emitError(
        "expected a permutation, but got a mapping that was not a "
        "permutation.");
    Diagnostic &note = diag.attachNote() << "mapping was:\n";
    printPermutation(permutation, note);
    return diag;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "PermuteOp produces underlying permutation: ";
    printPermutation(permutation, llvm::dbgs());
  });

  FrozenVector<int64_t> permKey = FrozenVector<int64_t>(std::move(permutation));
  ArrayRef<RotationGroup> rotationGroup =
      shiftNetworks.computeShiftNetwork(permKey);
  assert(!rotationGroup.empty() &&
         "Shift network must have at least one group");

  // Process each rotation group separately with a full set of power-of-two
  // shifts. Then sum the results together.
  rewriter.setInsertionPointAfter(op);
  std::optional<Value> result = std::nullopt;
  [[maybe_unused]] int groupIndex = 0;
  for (const RotationGroup &group : rotationGroup) {
    LLVM_DEBUG(llvm::dbgs()
               << "Implementing rotations for group " << groupIndex++ << "\n");
    Value perGroupResult =
        rotateGroup(op.getInput(), group, ciphertextSize, permKey, rewriter);
    if (result.has_value())
      result =
          rewriter.create<arith::AddIOp>(op.getLoc(), *result, perGroupResult);
    else
      result = perGroupResult;
  }

  rewriter.replaceOp(op, result.value());
  return success();
}

struct ImplementShiftNetwork
    : impl::ImplementShiftNetworkBase<ImplementShiftNetwork> {
  using ImplementShiftNetworkBase::ImplementShiftNetworkBase;

  void runOnOperation() override {
    VosVosErkinShiftNetworks shiftNetworks{ciphertextSize};

    getOperation()->walk([&](PermuteOp op) {
      if (failed(convertPermuteOp(op, shiftNetworks, ciphertextSize))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
