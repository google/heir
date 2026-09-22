#ifndef LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// The alignment map of an operator.
// lhsToRhs[d] is the rhs dim that lhs dim d aligns with, or kRepeatedDim when
// the rhs holds replication there. rhsToLhs is the reverse.
struct OperatorAlignmentMap {
  static constexpr int64_t kRepeatedDim = -1;
  SmallVector<int64_t> lhsToRhs;
  SmallVector<int64_t> rhsToLhs;

  static OperatorAlignmentMap identity(int64_t rank);
  // The matmul alignment map for the given operand ranks. The last lhs dim
  // and the second-to-last rhs dim (dim 0 for a vector) are the contraction dim
  // and align with each other; the free dims align with replication; leading
  // batch dims align from the right, and a batch dim with no partner aligns
  // with replication.
  static OperatorAlignmentMap matmul(int64_t lhsRank = 2, int64_t rhsRank = 2);
};

// Determines if two layouts are aligned for the operator.
bool isOperatorAligned(const OperatorAlignmentMap& map, LayoutAttr lhs,
                       LayoutAttr rhs);

// Determines the rolls each side must add to become aligned. Each added roll is
// a ROLL kernel. Fails when the dims are not aligned.
struct RollAlignment {
  llvm::SmallVector<RollSpec> addToLhs;
  llvm::SmallVector<RollSpec> addToRhs;
  bool empty() const { return addToLhs.empty() && addToRhs.empty(); }
};
FailureOr<RollAlignment> alignRolls(const OperatorAlignmentMap& map,
                                    LayoutAttr lhs, LayoutAttr rhs);

// Determines the piece list each side needs to match the other side's piece
// order: `forRhs` is a piece list for the rhs tensor in the lhs's order, and
// `forLhs` the reverse.
struct AlignedDims {
  llvm::SmallVector<DimAttr> forRhs;
  llvm::SmallVector<DimAttr> forLhs;
};
std::optional<AlignedDims> alignedDims(const OperatorAlignmentMap& map,
                                       ArrayRef<DimAttr> lhsDims,
                                       ArrayRef<DimAttr> rhsDims,
                                       MLIRContext* ctx);

// Grows each side to the operator's aligned shape by padding the smaller side
// with gap ciphertexts, turns gaps into replication right to left, and adds
// any missing factor as a new outermost replication piece. Fails when the two
// sides cannot both reach the shape.
struct ReplicatedPair {
  LayoutAttr lhs;
  LayoutAttr rhs;
};
FailureOr<ReplicatedPair> replicateForAlignment(const OperatorAlignmentMap& map,
                                                LayoutAttr lhs, LayoutAttr rhs,
                                                ArrayRef<int64_t> lhsShape,
                                                ArrayRef<int64_t> rhsShape);

struct AlignedPair {
  LayoutAttr lhs;
  LayoutAttr rhs;
};

// Applies the roll for swapping the summation dim to the ciphertext dims.
std::optional<LayoutAttr> applySumRoll(LayoutAttr layout, int64_t sumDim);

// Aligns with a roll.
llvm::SmallVector<AlignedPair, 2> rollToAlign(const OperatorAlignmentMap& map,
                                              LayoutAttr lhs, LayoutAttr rhs);

// Repacks one side into the layout the other side needs, rolls included.
std::optional<LayoutAttr> matchPublicLayout(const OperatorAlignmentMap& map,
                                            LayoutAttr lhs, LayoutAttr rhs,
                                            bool matchLhs);

llvm::SmallVector<AlignedPair> alignPair(const OperatorAlignmentMap& map,
                                         LayoutAttr lhs, LayoutAttr rhs);

// Computes the result layout of an aligned pair.
std::optional<LayoutAttr> outputLayout(const OperatorAlignmentMap& map,
                                       bool isMatmul, LayoutAttr lhs,
                                       LayoutAttr rhs, int64_t lhsSumDim,
                                       int64_t rhsSumDim);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
