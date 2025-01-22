#include "lib/Dialect/TensorExt/Transforms/LayoutConversionToShiftNetwork.h"

#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_NAME "layout-conversion-to-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_LAYOUTCONVERSIONTOSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

// A permutation of 0..n-1. This array ref should always have size n
// and contain each integer from 0 to n-1 exactly once.
// FIXME: replace this with a hashable type that can be used with DenseMap
using Permutation = SmallVector<int64_t>;

// A group of indices to rotate together
using RotationGroup = DenseSet<int64_t>;

inline Permutation identity(int64_t n) {
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
  ArrayRef<RotationGroup> computeShiftNetwork(Permutation permutation) {
    if (rotationGroups.count(permutation)) {
      return rotationGroups[permutation];
    }

    Permutation identityPerm = identity(ciphertextSize);
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
    rotationGroups.reserve(10);
    for (const auto &entry : coloring) {
      int64_t index = entry.first;
      int64_t color = entry.second;
      if (color >= rotationGroups.size()) {
        resultRotationGroups.resize(color + 1);
      }
      resultRotationGroups[color].insert(index);
    }

    rotationGroups[permutation] = resultRotationGroups;
    return rotationGroups[permutation];
  }

 private:
  int64_t ciphertextSize;
  DenseMap<Permutation, llvm::SmallVector<RotationGroup>> rotationGroups;
};

struct RewriteLayoutConversion : public OpRewritePattern<ConvertLayoutOp> {
  RewriteLayoutConversion(mlir::MLIRContext *context,
                          VosVosErkinShiftNetworks shiftNetworks)
      : mlir::OpRewritePattern<ConvertLayoutOp>(context),
        shiftNetworks(std::move(shiftNetworks)) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    return success();
  }

 private:
  VosVosErkinShiftNetworks shiftNetworks;
};

struct LayoutConversionToShiftNetwork
    : impl::LayoutConversionToShiftNetworkBase<LayoutConversionToShiftNetwork> {
  using LayoutConversionToShiftNetworkBase::LayoutConversionToShiftNetworkBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    VosVosErkinShiftNetworks shiftNetworks{ciphertextSize};
    patterns.add<RewriteLayoutConversion>(context, shiftNetworks);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
