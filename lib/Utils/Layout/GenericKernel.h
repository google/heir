#ifndef LIB_UTILS_LAYOUT_GENERICKERNEL_H_
#define LIB_UTILS_LAYOUT_GENERICKERNEL_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
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
//
// With a slot period `period` < numSlots of the output layout (see
// getSlotPeriod), the output copies of an element in one ciphertext fall into
// classes of slots congruent mod `period`, and only one copy per class is an
// instance: the one at or before the first position of the first secret input
// the iteration reads from a placed element, so that input's shift is below
// `period`. The groups then hold each class's value in one of its slots, and
// a rotate-and-reduce over the multiples of `period` completes the result.
struct InstanceGroups {
  // The distinct cleartext reads (row-major indices, one per cleartext input)
  // of a key's instances at one output slot, with their multiplicities.
  using Reads = std::vector<std::pair<SmallVector<int64_t>, int64_t>>;
  std::map<std::vector<int64_t>, std::map<int64_t, Reads>> readsByKey;
  // The slot period the instances were taken with.
  int64_t period;
  // Per output ciphertext and slot: the number of instances writing it,
  // counted after the rotate-and-reduce (in every copy of an element).
  std::vector<std::vector<int64_t>> coverage;
  // Per output ciphertext: whether it holds an output element no instance
  // writes.
  std::vector<bool> writesPartially;
};

struct GroupingOptions {
  // A slot period of the output layout, or numSlots for one instance per
  // output copy.
  int64_t period;
  // Stops the walk once the groups need more rotations of secret inputs.
  std::optional<int64_t> maxRotations;
};

// Groups the instances of `spec` by walking its loop nest. Fails if the output
// needs more than `numCts` ciphertexts or a slot is written more often than
// `kind` allows, and returns std::nullopt once the groups exceed
// `options.maxRotations`.
FailureOr<std::optional<InstanceGroups>> groupInstances(
    const GenericKernelSpec& spec, int64_t numCts, GenericReductionKind kind,
    const GroupingOptions& options, std::string& reason);

// The least slot period of a layout: the least tau > 0 such that moving every
// position (ct, slot) to (ct, (slot + tau) mod numSlots) maps the layout onto
// itself, or numSlots if there is no smaller one. Divides numSlots.
int64_t getSlotPeriod(const presburger::IntegerRelation& layout,
                      int64_t numSlots);

// What a kernel built from `groups` costs, compared lexicographically: the
// rotations (of secret inputs, and of the rotate-and-reduce per output
// ciphertext under a period), then the number of groups (plaintext products).
struct GroupCost {
  int64_t rotations;
  int64_t groups;
  bool operator<=(const GroupCost& other) const {
    return std::tie(rotations, groups) <=
           std::tie(other.rotations, other.groups);
  }
};
GroupCost getGroupCost(const InstanceGroups& groups, int64_t numSlots);

// The number of rotations of a rotate-and-reduce over `steps` multiples of a
// period (see implementRotateAndReduceAccumulation).
int64_t getRotateAndReduceRotations(int64_t steps);

// Groups the instances of `spec` with one instance per output copy and, for
// sums (`kind` None or Add) whose output layout has a slot period, with one
// per class of copies, and returns the cheaper grouping. Fails as
// groupInstances does.
FailureOr<InstanceGroups> groupInstancesByCost(const GenericKernelSpec& spec,
                                               int64_t numCts,
                                               GenericReductionKind kind,
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
// one slot, or more than one basic map).
FailureOr<presburger::IntegerRelation> inferInducedOutputLayout(
    ArrayRef<int64_t> loopRanges, AffineMap inputMap,
    const presburger::IntegerRelation& inputLayout, AffineMap outputMap,
    ArrayRef<int64_t> outputShape, bool reduceToLexmin, std::string& reason);

// The number of ciphertexts a (bounded, nonempty) layout uses: one more than
// its largest ciphertext index.
int64_t getNumCiphertexts(const presburger::IntegerRelation& layout);

// A layout for a secret input of a structured op, from loop indices
// (`loopRanges`) through `inputMap` to its data, under which the rotation each
// iteration needs, (flat_in(inputMap(d)) - flat_out(outputMap(d))) mod
// `numSlots` with flat = ct * numSlots + slot, does not depend on the loops
// the input pairs with output dims. A result of `inputMap` that reads one
// output loop p with stride s is split into q = floor((i - o) / s), placed
// with the output layout's pitch of that dim, and the residue (i - o) mod s.
// The residues and the results that read only reductions number classes,
// placed where the output layout puts the output dims the input does not read,
// and beyond those in further copies of the output's span. The layout places
// each element the op reads once, in no more ciphertexts than the input and
// output layouts use.
//
// Returns std::nullopt when `inputLayout` already has this property, and fails
// (with `reason`) when the op or the layouts do not have the required form:
// linear input indices that each read at most one output loop, an output
// layout whose flat position is affine in the output index, and inputs that
// fit the output extents.
FailureOr<std::optional<presburger::IntegerRelation>> getPolyphaseInputLayout(
    ArrayRef<int64_t> loopRanges, AffineMap inputMap,
    const presburger::IntegerRelation& inputLayout, AffineMap outputMap,
    const presburger::IntegerRelation& outputLayout,
    ArrayRef<int64_t> outputShape, int64_t numSlots, std::string& reason);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_GENERICKERNEL_H_
