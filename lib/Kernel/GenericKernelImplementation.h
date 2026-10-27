#ifndef LIB_KERNEL_GENERICKERNELIMPLEMENTATION_H_
#define LIB_KERNEL_GENERICKERNELIMPLEMENTATION_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Layout/GenericKernel.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

// A layout-driven FHE kernel for structured (linalg.generic-style) ops.
//
// A structured op iterates over a loop domain and, at each point d, reads
// input i at data index f_i(d) and writes (or accumulates into) the output at
// f_out(d), with affine f's. A layout L maps data indices to (ct, slot).
// Composing both gives, per iteration, the slot (ct_o, slot_o) that receives
// the body's value and the slot (ct_i, slot_i) each secret input value lives
// in, so input i must be rotated left by shift_i = (slot_i - slot_o) mod
// numSlots. Iterations with identical (ct_o, ct_i, shift_i)_i over the secret
// inputs form a group: one SIMD evaluation of the body on rotated inputs, times
// a plaintext mask of the slots the group writes (counting multiplicity for
// additive reductions), combined into out[ct_o]. See InstanceGroups for
// replicated and unplaced elements.
//
// Cleartext inputs need no layout: each group gathers, into every slot, the
// element that slot's iteration reads. The number of groups is the cost; a
// matvec with a cleartext matrix and a row-major vector yields one all-ones
// group per diagonal, i.e. the Halevi-Shoup kernel.
namespace mlir {
namespace heir {
namespace kernel {

// What the init of a structured op contributes to the ciphertexts it writes.
enum class GenericInitKind {
  // The init has no defined contents (tensor.empty).
  Empty,
  // The init is the identity of the combination: zero, or one for
  // multiplicative reductions.
  Neutral,
  // Anything else.
  Other,
};

// Builds the value one group computes in every slot, given its input
// ciphertexts and, when the body reads it, the init ciphertext (else null). For
// reductions this is the value combined into the accumulator, without the
// combining op itself.
template <typename T>
using GenericBodyBuilder = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    const std::vector<std::shared_ptr<ArithmeticDagNode<T>>>& inputs,
    const std::shared_ptr<ArithmeticDagNode<T>>& init)>;

// The combiner of an idempotent reduction (see GenericReductionKind).
template <typename T>
using GenericCombiner = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    const std::shared_ptr<ArithmeticDagNode<T>>& acc,
    const std::shared_ptr<ArithmeticDagNode<T>>& term)>;

struct GenericKernelConfig {
  GenericReductionKind kind = GenericReductionKind::None;
  GenericInitKind initKind = GenericInitKind::Empty;
  // Whether the body reads the init (only without reductions).
  bool bodyReadsInit = false;
  // The number of ciphertexts of the result.
  int64_t numCts = 1;
  // The element types of the result and of each input.
  DagType elementType;
  std::vector<DagType> inputElementTypes;
};

inline DagType tensorOfElementType(const DagType& elementType,
                                   std::vector<int64_t> shape) {
  if (const auto* intType = std::get_if<IntegerType>(&elementType.type_variant))
    return DagType::intTensor(intType->bitWidth, std::move(shape));
  const auto* floatType = std::get_if<FloatType>(&elementType.type_variant);
  assert(floatType && "expected an integer or float element type");
  return DagType::floatTensor(floatType->bitWidth, std::move(shape));
}

// Combines `values` pairwise into a balanced tree, which keeps the depth of
// the DAG (and of recursive visitors) logarithmic in the number of values.
template <typename NodePtr, typename CombineFn>
NodePtr combineBalanced(std::vector<NodePtr> values, CombineFn combine) {
  if (values.empty()) return nullptr;
  while (values.size() > 1) {
    std::vector<NodePtr> next;
    for (size_t i = 0; i + 1 < values.size(); i += 2) {
      next.push_back(combine(values[i], values[i + 1]));
    }
    if (values.size() % 2 == 1) next.push_back(values.back());
    values = std::move(next);
  }
  return values.front();
}

// Whether `node` is zero in every slot in which all of `zeros` are zero (with
// no `zeros`: whether it is zero everywhere). Conservative: opaque ops are
// assumed nonzero.
template <typename T>
bool vanishesWith(const std::shared_ptr<ArithmeticDagNode<T>>& node,
                  const std::set<const ArithmeticDagNode<T>*>& zeros) {
  if (zeros.count(node.get())) return true;
  return std::visit(
      [&](const auto& n) {
        using NodeType = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<NodeType, AddNode<T>> ||
                      std::is_same_v<NodeType, SubtractNode<T>>) {
          return vanishesWith(n.left, zeros) && vanishesWith(n.right, zeros);
        } else if constexpr (std::is_same_v<NodeType, MultiplyNode<T>>) {
          return vanishesWith(n.left, zeros) || vanishesWith(n.right, zeros);
        } else if constexpr (std::is_same_v<NodeType, ConstantTensorNode>) {
          return llvm::all_of(n.value, [](double v) { return v == 0; });
        } else if constexpr (std::is_same_v<NodeType, ConstantScalarNode> ||
                             std::is_same_v<NodeType, SplatNode>) {
          return n.value == 0;
        } else {
          return false;
        }
      },
      node->node_variant);
}

// Builds the kernel of a structured op as a DAG over `inputs`: per secret
// input its packed ciphertexts (numCts x numSlots), per cleartext input its
// data tensor (null for an iota, which linalg.index reads). `init` is the
// packed init in the output layout (null if it is empty and unread). Returns
// the packed result of shape (config.numCts x spec.numSlots), or fails with
// `reason`.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 FailureOr<std::shared_ptr<ArithmeticDagNode<T>>>>
implementGenericKernel(const GenericKernelSpec& spec,
                       ArrayRef<std::shared_ptr<ArithmeticDagNode<T>>> inputs,
                       const std::shared_ptr<ArithmeticDagNode<T>>& init,
                       const GenericKernelConfig& config,
                       const GenericBodyBuilder<T>& body,
                       const GenericCombiner<T>& idempotentCombine,
                       std::string& reason) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;
  assert(inputs.size() == spec.inputs.size());
  const int64_t numSlots = spec.numSlots;
  const int64_t numCts = config.numCts;
  const bool isMul = config.kind == GenericReductionKind::Mul;
  const bool isIdempotent = config.kind == GenericReductionKind::Idempotent;
  if (isIdempotent && config.initKind == GenericInitKind::Empty) {
    reason = "an idempotent reduction needs an init to start from";
    return failure();
  }

  FailureOr<InstanceGroups> groups =
      groupInstances(spec, numCts, config.kind, reason);
  if (failed(groups)) return failure();
  const auto& readsByKey = groups->readsByKey;
  const auto& coverage = groups->coverage;

  const DagType ctType = tensorOfElementType(config.elementType, {1, numSlots});
  auto constant = [&](const std::vector<int64_t>& values, const DagType& type) {
    return NodeTy::constantTensor(
        std::vector<double>(values.begin(), values.end()), type);
  };
  // Masks in the result type, shared between groups that write the same slots.
  std::map<std::vector<int64_t>, NodePtr> masks;
  auto getMask = [&](const std::vector<int64_t>& values) {
    NodePtr& result = masks[values];
    if (!result) result = constant(values, ctType);
    return result;
  };
  auto zero = [&](const DagType& elementType) {
    return constant(std::vector<int64_t>(numSlots, 0),
                    tensorOfElementType(elementType, {1, numSlots}));
  };
  // Shared nodes, so each extraction, rotation and gather is built once.
  std::map<std::pair<size_t, int64_t>, NodePtr> extracted;
  std::map<std::tuple<size_t, int64_t, int64_t>, NodePtr> rotated;
  auto getRotated = [&](size_t input, int64_t ct, int64_t shift) {
    NodePtr& result = rotated[{input, ct, shift}];
    if (result) return result;
    NodePtr& ciphertext = extracted[{input, ct}];
    if (!ciphertext) {
      ciphertext = NodeTy::extract(inputs[input], static_cast<size_t>(ct));
    }
    result = shift == 0 ? ciphertext : NodeTy::leftRotate(ciphertext, shift);
    return result;
  };
  std::map<std::pair<size_t, std::vector<int64_t>>, NodePtr> gathered;
  auto getGathered = [&](size_t input, const std::vector<int64_t>& indices) {
    NodePtr& result = gathered[{input, indices}];
    if (!result) result = NodeTy::gather(inputs[input], indices, {1, numSlots});
    return result;
  };
  std::map<int64_t, NodePtr> initCiphertexts;
  auto getInit = [&](int64_t ct) {
    assert(init && "the init is read but was not given");
    NodePtr& result = initCiphertexts[ct];
    if (!result) result = NodeTy::extract(init, static_cast<size_t>(ct));
    return result;
  };
  auto combine = [&](NodePtr lhs, NodePtr rhs) {
    return isMul ? NodeTy::mul(lhs, rhs) : NodeTy::add(lhs, rhs);
  };

  // A key yields one group per r: group r reads the r-th distinct cleartext
  // read of every slot, so no slot needs two cleartext values at once.
  std::vector<std::vector<NodePtr>> terms(numCts);
  std::map<int64_t, NodePtr> idempotentAcc;
  for (const auto& [key, slots] : readsByKey) {
    const int64_t outputCt = key[0];
    size_t numGroups = 0;
    for (const auto& [slot, reads] : slots) {
      numGroups = std::max(numGroups, reads.size());
    }
    for (size_t r = 0; r < numGroups; ++r) {
      std::vector<int64_t> mask(numSlots, 0);
      std::vector<std::vector<int64_t>> rows;
      for (const auto& [slot, reads] : slots) {
        if (r >= reads.size()) continue;
        mask[slot] = reads[r].second;
        rows.resize(reads[r].first.size(), std::vector<int64_t>(numSlots, -1));
        for (auto [row, read] : llvm::zip(rows, reads[r].first)) {
          row[slot] = read;
        }
      }
      std::vector<NodePtr> args;
      // Arguments that are zero in the slots the group does not write.
      std::set<const NodeTy*> zeroOutsideGroup;
      for (size_t i = 0, secret = 1, cleartext = 0; i < inputs.size(); ++i) {
        if (spec.inputs[i].isCleartext()) {
          args.push_back(getGathered(i, rows[cleartext++]));
          zeroOutsideGroup.insert(args.back().get());
          continue;
        }
        args.push_back(key[secret] < 0
                           ? zero(config.inputElementTypes[i])
                           : getRotated(i, key[secret], key[secret + 1]));
        secret += 2;
      }
      NodePtr term =
          body(args, config.bodyReadsInit ? getInit(outputCt) : nullptr);
      std::vector<int64_t> written;
      for (int64_t m : mask) written.push_back(m > 0 ? 1 : 0);
      const bool writesAll = !llvm::is_contained(written, 0);
      if (isIdempotent) {
        // A group only updates the slots it writes:
        // acc + written * (combine(acc, term) - acc).
        NodePtr& acc = idempotentAcc[outputCt];
        if (!acc) acc = getInit(outputCt);
        NodePtr combined = idempotentCombine(acc, term);
        acc = writesAll
                  ? combined
                  : NodeTy::add(acc, NodeTy::mul(getMask(written),
                                                 NodeTy::sub(combined, acc)));
        continue;
      }
      // Sums skip terms that are zero, e.g. products with zero padding.
      if (!isMul && vanishesWith<T>(term, {})) continue;
      if (mask != std::vector<int64_t>(numSlots, 1)) {
        // A gathered cleartext is zero exactly where the group writes nothing,
        // so a cleartext factor masks the term already.
        bool masked = mask == written && vanishesWith(term, zeroOutsideGroup);
        if (!masked) term = NodeTy::mul(term, getMask(mask));
        if (isMul) {
          // Slots outside the group must multiply by one.
          std::vector<int64_t> complement;
          for (int64_t w : written) complement.push_back(1 - w);
          term = NodeTy::add(term, getMask(complement));
        }
      }
      terms[outputCt].push_back(term);
    }
  }

  NodePtr result = NodeTy::empty(
      {numCts, numSlots},
      tensorOfElementType(config.elementType, {numCts, numSlots}));
  for (int64_t ct = 0; ct < numCts; ++ct) {
    const bool isWritten =
        llvm::any_of(coverage[ct], [](int64_t c) { return c > 0; });
    NodePtr acc = isIdempotent ? idempotentAcc[ct]
                               : combineBalanced(std::move(terms[ct]), combine);
    // Every term of a written ciphertext may have vanished.
    if (!acc && isWritten && !isIdempotent) acc = zero(config.elementType);
    if (!acc) {
      // An unwritten ciphertext is the init.
      if (config.initKind == GenericInitKind::Empty) continue;
      acc = getInit(ct);
    } else if (config.initKind == GenericInitKind::Other && !isIdempotent) {
      if (config.kind != GenericReductionKind::None) {
        // Reductions accumulate into the init.
        acc = combine(acc, getInit(ct));
      } else {
        // Partial writes keep the init's untouched slots.
        std::vector<int64_t> keep;
        for (int64_t c : coverage[ct]) keep.push_back(c == 0 ? 1 : 0);
        if (llvm::is_contained(keep, 1)) {
          acc = NodeTy::add(acc, NodeTy::mul(getInit(ct), getMask(keep)));
        }
      }
    }
    result = NodeTy::insert(acc, result, static_cast<size_t>(ct));
  }
  return result;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_GENERICKERNELIMPLEMENTATION_H_
