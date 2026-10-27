#ifndef LIB_UTILS_LAYOUT_GENERICKERNEL_H_
#define LIB_UTILS_LAYOUT_GENERICKERNEL_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

// Layout utilities for the generic kernel of structured (linalg.generic-style)
// ops, see lib/Kernel/GenericKernelImplementation.h.
namespace mlir {
namespace heir {

// An operand: its indexing map (loop indices -> data indices) and, for a
// secret operand, the layout of its data tensor (data indices -> (ct, slot)).
struct StructuredOperand {
  AffineMap indexingMap;
  presburger::IntegerRelation layout{
      presburger::PresburgerSpace::getRelationSpace(0, 2)};
  // The data shape of a cleartext operand, which has no layout.
  std::optional<SmallVector<int64_t>> cleartextShape;

  bool isCleartext() const { return cleartextShape.has_value(); }
};

struct GenericKernelSpec {
  // Loop index j ranges over [0, loopRanges[j]).
  SmallVector<int64_t> loopRanges;
  std::vector<StructuredOperand> inputs;
  // The output always has a layout.
  StructuredOperand output;
  int64_t numSlots;
};

// Checks that the operand layouts map their data to (ct, slot) and that the
// output layout places distinct output elements in distinct slots.
LogicalResult validateGenericKernelSpec(const GenericKernelSpec& spec,
                                        std::string& reason);

// How a reduction combines into its accumulator. Idempotent stands for an
// associative, commutative and idempotent combiner (max, min, and, or), for
// which repeated terms do not change the result.
enum class GenericReductionKind { None, Add, Mul, Idempotent };

// The instances of a structured op, (iteration, output slot) pairs, grouped by
// key [ct_o, (ct_i, shift_i) per secret input]: the output ciphertext and, per
// secret input, the ciphertext holding the element the instance reads and the
// rotation that moves it to the output slot. A replicated element takes its
// lexicographically smallest (ct_i, shift_i), which keeps periodic replicas in
// one group; ct_i = -1 marks an element the layout does not place (read as
// zero).
struct InstanceGroups {
  // The distinct cleartext reads (row-major indices, one per cleartext input)
  // of a key's instances at one output slot, with their multiplicities.
  using Reads = std::vector<std::pair<SmallVector<int64_t>, int64_t>>;
  std::map<std::vector<int64_t>, std::map<int64_t, Reads>> readsByKey;
  // Per output ciphertext and slot: the number of instances writing it.
  std::vector<std::vector<int64_t>> coverage;
};

// Groups the instances of `spec` by walking its loop nest. Fails if the output
// needs more than `numCts` ciphertexts or a slot is written more often than
// `kind` allows.
FailureOr<InstanceGroups> groupInstances(const GenericKernelSpec& spec,
                                         int64_t numCts,
                                         GenericReductionKind kind,
                                         std::string& reason);

// The number of keys of the instances of a structured op (see
// InstanceGroups), a lower bound on the number of groups (a key yields one
// group per distinct cleartext read of a slot), and the number of distinct
// nonzero rotations of the secret inputs.
struct GroupCount {
  int64_t numKeys = 0;
  int64_t numRotations = 0;
};

// Counts the keys of groupInstances without walking the loop nest: the
// smallest rotation over the replicas of each input element is one ISL lexmin
// over all instances, and only the distinct keys are enumerated. Fails when
// ISL exceeds `maxOperations` or there are more than `maxKeys` keys.
FailureOr<GroupCount> countInstanceGroups(const GenericKernelSpec& spec,
                                          int64_t maxKeys,
                                          unsigned long maxOperations,
                                          std::string& reason);

// Evaluates a symbol-free affine map at a point.
SmallVector<int64_t> evaluateAffineMap(AffineMap map, ArrayRef<int64_t> point);

// The relation from loop indices (bounded by `loopRanges`) to data indices of
// an affine indexing map.
FailureOr<presburger::IntegerRelation> getIndexingMapRelation(
    AffineMap map, ArrayRef<int64_t> loopRanges);

// Whether the output indexing map writes every element of a tensor of the
// given shape, as opposed to a partial write such as a concatenation piece.
bool outputMapCoversResult(AffineMap outputMap, ArrayRef<int64_t> loopRanges,
                           ArrayRef<int64_t> outputShape);

// The output layout induced by an input: output element f_out(d) is placed in
// the slots of input element f_in(d). With `reduceToLexmin` (reductions, where
// many d share an output element) each output element gets the
// lexicographically smallest such (ct, slot). Fails, with `reason`, when the
// result is not a valid layout (an element without a slot, two elements in
// one slot, or more than one basic map), or after `maxOperations` ISL
// operations.
FailureOr<presburger::IntegerRelation> inferInducedOutputLayout(
    ArrayRef<int64_t> loopRanges, AffineMap inputMap,
    const presburger::IntegerRelation& inputLayout, AffineMap outputMap,
    ArrayRef<int64_t> outputShape, bool reduceToLexmin,
    unsigned long maxOperations, std::string& reason);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_GENERICKERNEL_H_
