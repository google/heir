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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Layout/GenericKernel.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/Hashing.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/ScopeExit.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

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
  // The init is the identity of the combination: zero, one for multiplicative
  // reductions, or e.g. -inf for max.
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

  FailureOr<InstanceGroups> groups =
      groupInstances(spec, numCts, config.kind, reason);
  if (failed(groups)) return failure();
  if (isIdempotent && config.initKind == GenericInitKind::Neutral &&
      llvm::is_contained(groups->writesPartially, true)) {
    reason =
        "an idempotent reduction with an identity init must write every "
        "output element";
    return failure();
  }
  const auto& readsByKey = groups->readsByKey;
  const auto& coverage = groups->coverage;

  const DagType ctType = tensorOfElementType(config.elementType, {1, numSlots});
  auto constant = [&](const std::vector<int64_t>& values, const DagType& type) {
    return NodeTy::constantTensor(
        std::vector<double>(values.begin(), values.end()), type);
  };
  // Masks in the result type, shared between groups that write the same slots.
  std::unordered_map<size_t, SmallVector<NodePtr, 1>> masksByHash;
  auto getMask = [&](const std::vector<int64_t>& values) {
    llvm::hash_code h = llvm::hash_value(0);
    for (size_t idx = 0; idx < values.size(); ++idx) {
      if (values[idx] != 0) h = llvm::hash_combine(h, idx, values[idx]);
    }
    auto& bucket = masksByHash[static_cast<size_t>(h)];
    for (const NodePtr& cand : bucket) {
      const auto& cNode = std::get<ConstantTensorNode>(cand->node_variant);
      if (cNode.value.size() == values.size() &&
          std::equal(values.begin(), values.end(), cNode.value.begin())) {
        return cand;
      }
    }
    NodePtr created = constant(values, ctType);
    bucket.push_back(created);
    return created;
  };
  auto makeZero = [&](const DagType& elementType) {
    return constant(std::vector<int64_t>(numSlots, 0),
                    tensorOfElementType(elementType, {1, numSlots}));
  };
  NodePtr outputZero;
  auto getOutputZero = [&]() {
    if (!outputZero) outputZero = makeZero(config.elementType);
    return outputZero;
  };
  std::vector<NodePtr> inputZeros(inputs.size());
  auto getInputZero = [&](size_t i) {
    if (!inputZeros[i]) inputZeros[i] = makeZero(config.inputElementTypes[i]);
    return inputZeros[i];
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
  std::unordered_map<size_t, SmallVector<NodePtr, 1>> gatheredByHash;
  auto getGathered = [&](size_t input, const std::vector<int64_t>& indices,
                         size_t rowHash) {
    size_t h = static_cast<size_t>(llvm::hash_combine(input, rowHash));
    auto& bucket = gatheredByHash[h];
    for (const NodePtr& cand : bucket) {
      const auto& gNode = std::get<GatherNode<T>>(cand->node_variant);
      if (gNode.source == inputs[input] && gNode.indices == indices) {
        return cand;
      }
    }
    NodePtr created = NodeTy::gather(inputs[input], indices, {1, numSlots});
    bucket.push_back(created);
    return created;
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
  const size_t numCleartextInputs = llvm::count_if(
      spec.inputs, [](const auto& input) { return input.isCleartext(); });
  std::vector<std::vector<NodePtr>> terms(numCts);
  std::map<int64_t, NodePtr> idempotentAcc;
  // Per output ciphertext, the slots an idempotent reduction has written.
  std::map<int64_t, std::vector<int64_t>> seenSlots;
  std::vector<int64_t> mask(numSlots, 0);
  std::vector<std::vector<int64_t>> rows(numCleartextInputs,
                                         std::vector<int64_t>(numSlots, -1));
  SmallVector<int64_t, 64> activeSlots;
  SmallVector<llvm::hash_code, 2> rowHashes(numCleartextInputs);

  for (const auto& [key, slots] : readsByKey) {
    const int64_t outputCt = key[0];
    size_t numGroups = 0;
    for (const auto& [slot, reads] : slots) {
      numGroups = std::max(numGroups, reads.size());
    }
    for (size_t r = 0; r < numGroups; ++r) {
      activeSlots.clear();
      bool anyMultNe1 = false;
      rowHashes.assign(numCleartextInputs, llvm::hash_value(0));
      for (const auto& [slot, reads] : slots) {
        if (r >= reads.size()) continue;
        int64_t mult = reads[r].second;
        assert(mult > 0 && "expected positive multiplicity");
        assert(reads[r].first.size() == numCleartextInputs &&
               "expected one read index per cleartext input");
        mask[slot] = mult;
        activeSlots.push_back(slot);
        if (mult != 1) anyMultNe1 = true;
        for (size_t c = 0; c < numCleartextInputs; ++c) {
          int64_t rd = reads[r].first[c];
          rows[c][slot] = rd;
          if (rd >= 0) {
            rowHashes[c] = llvm::hash_combine(rowHashes[c], slot, rd);
          }
        }
      }
      auto resetScratch = llvm::scope_exit([&]() {
        for (int64_t slot : activeSlots) {
          mask[slot] = 0;
          for (auto& row : rows) row[slot] = -1;
        }
      });
      std::vector<NodePtr> args;
      // Arguments that are zero in the slots the group does not write.
      std::set<const NodeTy*> zeroOutsideGroup;
      for (size_t i = 0, secret = 1, cleartext = 0; i < inputs.size(); ++i) {
        if (spec.inputs[i].isCleartext()) {
          args.push_back(getGathered(
              i, rows[cleartext], static_cast<size_t>(rowHashes[cleartext])));
          ++cleartext;
          zeroOutsideGroup.insert(args.back().get());
          continue;
        }
        args.push_back(key[secret] < 0
                           ? getInputZero(i)
                           : getRotated(i, key[secret], key[secret + 1]));
        secret += 2;
      }
      NodePtr term =
          body(args, config.bodyReadsInit ? getInit(outputCt) : nullptr);
      const bool writesAll =
          activeSlots.size() == static_cast<size_t>(numSlots);
      if (isIdempotent) {
        // A group sets the slots it writes first and combines into the slots
        // written before: acc + first * (term - acc) + later * (combine(acc,
        // term) - acc). An infinite identity (as for max) is thus never read;
        // any other init counts as written.
        const bool readsInit = config.initKind == GenericInitKind::Other;
        NodePtr& acc = idempotentAcc[outputCt];
        auto [seen, unused] =
            seenSlots.try_emplace(outputCt, numSlots, readsInit ? 1 : 0);
        if (!acc && readsInit) acc = getInit(outputCt);
        std::vector<int64_t> first(numSlots, 0), later(numSlots, 0);
        for (int64_t slot : activeSlots) {
          (seen->second[slot] ? later : first)[slot] = 1;
          seen->second[slot] = 1;
        }
        if (!acc) {
          acc = writesAll ? term : NodeTy::mul(term, getMask(first));
          continue;
        }
        NodePtr next = acc;
        if (llvm::is_contained(later, 1)) {
          NodePtr combined = idempotentCombine(acc, term);
          next =
              !llvm::is_contained(later, 0)
                  ? combined
                  : NodeTy::add(next, NodeTy::mul(getMask(later),
                                                  NodeTy::sub(combined, acc)));
        }
        if (llvm::is_contained(first, 1)) {
          next = NodeTy::add(
              next, NodeTy::mul(getMask(first), NodeTy::sub(term, acc)));
        }
        acc = next;
        continue;
      }
      // Sums skip terms that are zero, e.g. products with zero padding.
      if (!isMul && vanishesWith<T>(term, {})) continue;
      if (!writesAll || anyMultNe1) {
        // A gathered cleartext is zero exactly where the group writes nothing,
        // so a cleartext factor masks the term already.
        bool masked = !anyMultNe1 && vanishesWith(term, zeroOutsideGroup);
        if (!masked) term = NodeTy::mul(term, getMask(mask));
        if (isMul) {
          // Slots outside the group must multiply by one.
          std::vector<int64_t> complement(numSlots, 1);
          for (int64_t slot : activeSlots) complement[slot] = 0;
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
    if (!acc && isWritten && !isIdempotent) acc = getOutputZero();
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
