#include "lib/Dialect/TensorExt/Transforms/LayoutConversionToShiftNetwork.h"

#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

#define DEBUG_TYPE "layout-conversion-to-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_LAYOUTCONVERSIONTOSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

// A permutation of 0..n-1. This vector should always have size n and contain
// each integer from 0 to n-1 exactly once.
using Permutation = FrozenVector<int64_t>;

// A group of indices to rotate together
using RotationGroup = DenseSet<int64_t>;

inline SmallVector<int64_t> identity(int64_t n) {
  SmallVector<int64_t> permutation;
  for (int64_t i = 0; i < n; i++) {
    permutation.push_back(i);
  }
  return permutation;
}

// Cf. https://www.jeremykun.com/2024/09/02/shift-networks/
// and https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20
// for an explanation of the algorithm.
class VosVosErkinShiftNetworks {
 public:
  VosVosErkinShiftNetworks(int64_t ciphertextSize)
      : ciphertextSize(ciphertextSize) {}

  // Computes the shift network for a given permutation of ciphertext indices.
  // The returned ArrayRef is owned by this VosVosErkinShiftNetworks object.
  // The resulting shift is cached, and the cache is used on further calls to
  // avoid recomputing the shift network.
  ArrayRef<RotationGroup> computeShiftNetwork(const Permutation &permutation) {
    if (rotationGroups.count(permutation)) {
      return rotationGroups[permutation];
    }

    // Stores the amount that each ciphertext index is shifted forward.
    SmallVector<int64_t> shifts;
    for (int64_t i = 0; i < ciphertextSize; i++) {
      shifts.push_back((permutation[i] - i) % ciphertextSize);
    }

    // We apply power-of-two shifts to each set bit in LSB-to-MSB order, 1, 2,
    // 4, 8, ..., and identify conflicts that would occur. Each shift amount is
    // considered a "round" in which a group of indices are attempted to be
    // shifted together.
    SmallVector<SmallVector<int64_t>> rounds;
    // The identity permutation is a base case where no shifts are applied.
    Permutation identityPerm = FrozenVector<int64_t>(identity(ciphertextSize));
    for (int64_t rotationAmount = 1; rotationAmount <= ciphertextSize;
         rotationAmount <<= 1) {
      SmallVector<int64_t> round;
      ArrayRef<int64_t> lastRound =
          !rounds.empty() ? ArrayRef<int64_t>(rounds.back()) : identityPerm;
      int inputIndex = 0;
      for (int64_t shift : shifts) {
        // The bit is set, implying we would rotate by 2**bit in this round
        if (shift & rotationAmount) {
          round.push_back(lastRound[inputIndex] + rotationAmount);
        } else {
          // Otherwise the value is unchanged from last round
          round.push_back(lastRound[inputIndex]);
        }
        ++inputIndex;
      }
      rounds.push_back(round);
    }

    // Create a graph whose vertices are the input indices to permute, and
    // whose edges are conflicts: an edge being present means the two indices
    // cannot participate in the same rotation group.
    graph::UndirectedGraph<int64_t> conflictGraph;
    for (int64_t i = 0; i < ciphertextSize; i++) {
      conflictGraph.addVertex(i);
    }
    for (const SmallVector<int64_t> &round : rounds) {
      for (int64_t i = 0; i < ciphertextSize; i++) {
        for (int64_t j = i + 1; j < ciphertextSize; j++) {
          if (round[i] == round[j]) {
            conflictGraph.addEdge(i, j);
          }
        }
      }
    }

    graph::GreedyGraphColoring<int64_t> colorer;
    std::unordered_map<int64_t, int> coloring = colorer.color(conflictGraph);

    SmallVector<RotationGroup> resultRotationGroups;
    rotationGroups.reserve(64);
    for (const auto &entry : coloring) {
      int64_t index = entry.first;
      int64_t color = entry.second;
      if (color >= rotationGroups.size()) {
        resultRotationGroups.resize(color + 1);
      }
      resultRotationGroups[color].insert(index);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Splitting permutation into permutation groups:\n";
      for (int i = 0; i < resultRotationGroups.size(); i++) {
        llvm::dbgs() << "Group " << i << ": ";
        for (int64_t index : resultRotationGroups[i]) {
          llvm::dbgs() << index << " ";
        }
        llvm::dbgs() << "\n";
      }
    });

    rotationGroups[permutation] = resultRotationGroups;
    return rotationGroups[permutation];
  }

  int64_t getCiphertextSize() const { return ciphertextSize; }

 private:
  int64_t ciphertextSize;
  DenseMap<Permutation, llvm::SmallVector<RotationGroup>> rotationGroups;
};

// Create a tensor with zeros everywhere except for the indices specified in
// the input `indices` vector.
Value createMask(TypedValue<RankedTensorType> tensor,
                 const SmallVector<int64_t> &indices, IRRewriter &rewriter) {
  auto elementType = tensor.getType().getElementType();
  SmallVector<Attribute> maskAttrs;

  for (int64_t i = 0; i < tensor.getType().getDimSize(0); i++) {
    maskAttrs.push_back(rewriter.getIntegerAttr(elementType, 0));
  }
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
                  IRRewriter &rewriter) {
  std::optional<Value> result = std::nullopt;
  // As we rotate indices by partial shifts, we need to keep track of where
  // each index currently is in the tensor.
  DenseMap<int64_t, int64_t> inputIndexToCurrentPosition;
  inputIndexToCurrentPosition.reserve(group.size());
  for (int64_t index : group) {
    inputIndexToCurrentPosition[index] = index;
  }

  for (int64_t rotationAmount = 1; rotationAmount <= ciphertextSize;
       rotationAmount <<= 1) {
    SmallVector<int64_t> indicesToRotate;
    for (int64_t index : group) {
      if (index & rotationAmount) {
        indicesToRotate.push_back(inputIndexToCurrentPosition[index]);
      }
    }
    if (indicesToRotate.empty()) {
      continue;
    }

    Value mask = createMask(tensor, indicesToRotate, rewriter);
    arith::MulIOp maskOp =
        rewriter.create<arith::MulIOp>(tensor.getLoc(), tensor, mask);
    // rotating right, so negate the shift amount
    Value rotated = rewriter.create<tensor_ext::RotateOp>(
        tensor.getLoc(), maskOp.getResult(),
        rewriter.create<arith::ConstantIntOp>(tensor.getLoc(), -rotationAmount,
                                              rewriter.getI32Type()));

    if (result.has_value()) {
      result = rewriter.create<arith::AddIOp>(tensor.getLoc(), result.value(),
                                              rotated);
    } else {
      result = rotated;
    }

    for (auto index : indicesToRotate) {
      inputIndexToCurrentPosition[index] =
          (inputIndexToCurrentPosition[index] + rotationAmount) %
          ciphertextSize;
    }
  }

  return result.has_value() ? result.value() : tensor;
}

LogicalResult convertLayoutOp(ConvertLayoutOp op,
                              VosVosErkinShiftNetworks &shiftNetworks,
                              int64_t ciphertextSize) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  IRRewriter rewriter(op.getContext());
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  // Convert the input and output layouts to an explicit permutation.
  AffineMap inputLayout = op.getFromLayout().getValue();
  AffineMap outputLayout = op.getToLayout().getValue();

  // Only support a 1-D tensor
  if (op.getTensor().getType().getRank() != 1) {
    return op.emitError("requires a one-dimensional tensor");
  }

  // For now assume the layout maps have one result (single ciphertext)
  if (inputLayout.getNumResults() != 1 || outputLayout.getNumResults() != 1) {
    return op.emitError()
           << "Shift network lowering only supports layout affine_maps with "
              "a single result (i.e., one ciphertext).";
  }

  // FIXME: Should I simplify these to better facilitate the equality check?
  if (inputLayout == outputLayout) {
    // Just forward the operand
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }

  // The concrete permutation is the result of iterating over the index space
  // of the tensors, and mapping fromLayout.eval(index) to
  // toLayout.eval(index).
  ArrayRef<int64_t> dims = op.getTensor().getType().getShape();

  // Initial permutation starts as the identity permutation.
  // FIXME: start with an empty partial mapping, then extend it to a permutation
  // in some way?
  SmallVector<int64_t> permutation = identity(ciphertextSize);

  LLVM_DEBUG(llvm::dbgs() << "Constructing permutation...\n");
  // Looking for something like llvm::product_iterator, but found nothing.
  // Iterating manually and using mod arithmetic to get the per-axis indices.
  SmallVector<int64_t, 4> indices;
  indices.resize(dims.size());
  for (size_t index = 0; index < op.getTensor().getType().getNumElements();
       ++index) {
    // Unflatten the index into dimensional components
    int dimIndex = dims.size() - 1;
    int indexCopy = index;
    for (int64_t dim : llvm::reverse(dims)) {
      indices[dimIndex] = indexCopy % dim;
      indexCopy /= dim;
      --dimIndex;
    }

    SmallVector<Attribute> inputLayoutResults;
    SmallVector<Attribute> outputLayoutResults;
    SmallVector<Attribute> operandConstants;
    for (int64_t i = 0; i < dims.size(); i++) {
      operandConstants.push_back(rewriter.getI64IntegerAttr(indices[i]));
    }
    if (failed(
            inputLayout.constantFold(operandConstants, inputLayoutResults)) ||
        failed(
            outputLayout.constantFold(operandConstants, outputLayoutResults))) {
      return op.emitError(
          "unable to statically evaluate one of the two affine maps.");
    }

    int64_t inputLayoutResultIndex =
        cast<IntegerAttr>(inputLayoutResults[0]).getInt();
    int64_t outputLayoutResultIndex =
        cast<IntegerAttr>(outputLayoutResults[0]).getInt();
    permutation[inputLayoutResultIndex] = outputLayoutResultIndex;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "ConvertLayoutOp produces underlying permutation: ";
    for (int i = 0; i < permutation.size(); i++) {
      llvm::dbgs() << i << " -> " << permutation[i] << ", ";
      if (i % 10 == 9) {
        llvm::dbgs() << "\n";
      }
    }
    llvm::dbgs() << "\n";
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
  for (const RotationGroup &group : rotationGroup) {
    Value perGroupResult =
        rotateGroup(op.getTensor(), group, ciphertextSize, rewriter);
    if (result.has_value())
      result =
          rewriter.create<arith::AddIOp>(op.getLoc(), *result, perGroupResult);
    else
      result = perGroupResult;
  }

  rewriter.replaceOp(op, result.value());
  return success();
}

struct LayoutConversionToShiftNetwork
    : impl::LayoutConversionToShiftNetworkBase<LayoutConversionToShiftNetwork> {
  using LayoutConversionToShiftNetworkBase::LayoutConversionToShiftNetworkBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    VosVosErkinShiftNetworks shiftNetworks{ciphertextSize};

    getOperation()->walk([&](ConvertLayoutOp op) {
      if (failed(convertLayoutOp(op, shiftNetworks, ciphertextSize))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
