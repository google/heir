#include "lib/Transforms/LayoutOptimization/LayoutConversionCost.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/Hashing.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

#define DEBUG_TYPE "layout-conversion-cost"

namespace mlir {
namespace heir {

using kernel::AddNode;
using kernel::ArithmeticDagNode;
using kernel::CachingVisitor;
using kernel::ConstantScalarNode;
using kernel::ConstantTensorNode;
using kernel::LeafNode;
using kernel::LeftRotateNode;
using kernel::MultiplyNode;
using kernel::SubtractNode;
using kernel::SymbolicValue;
using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;
using tensor_ext::CtSlot;
using tensor_ext::Mapping;
using tensor_ext::NewLayoutAttr;
using tensor_ext::ShiftScheme;

// A visitor that counts the number of rotations in an ArithmeticDag.
class RotationCountVisitor : public CachingVisitor<SymbolicValue, int64_t> {
 public:
  using CachingVisitor<SymbolicValue, int64_t>::operator();

  RotationCountVisitor() : CachingVisitor<SymbolicValue, int64_t>() {}

  int64_t operator()(const ConstantScalarNode& node) override { return 0.0; }

  int64_t operator()(const ConstantTensorNode& node) override { return 0.0; }

  int64_t operator()(const LeafNode<SymbolicValue>& node) override {
    return 0.0;
  }

  int64_t operator()(const AddNode<SymbolicValue>& node) override {
    return this->process(node.left) + this->process(node.right);
  }

  int64_t operator()(const SubtractNode<SymbolicValue>& node) override {
    return this->process(node.left) + this->process(node.right);
  }

  int64_t operator()(const MultiplyNode<SymbolicValue>& node) override {
    return this->process(node.left) + this->process(node.right);
  }

  int64_t operator()(const LeftRotateNode<SymbolicValue>& node) override {
    return this->process(node.operand) + 1;
  }
};

Cost computeCostOfLayoutConversion(int64_t numCiphertexts,
                                   int64_t ciphertextSize,
                                   NewLayoutAttr fromLayout,
                                   NewLayoutAttr toLayout,
                                   std::size_t vveRandomSeed,
                                   unsigned vveRandomTries) {
  if (fromLayout == toLayout) {
    LLVM_DEBUG(llvm::dbgs() << "Layouts are the same, conversion cost is 0\n";);
    return 0;
  }

  std::shared_ptr<IntegerRelation> composedLayout =
      fromLayout.getIntegerRelation().clone();
  composedLayout->inverse();
  composedLayout->compose(toLayout.getIntegerRelation());

  Mapping mapping(ciphertextSize, numCiphertexts);
  PointPairCollector collector(2, 2);
  enumeratePoints(*composedLayout, collector);
  for (const auto& [source, target] : collector.points) {
    mapping.add(CtSlot{source[0], source[1]}, CtSlot{target[0], target[1]});
  }

  tensor_ext::VosVosErkinShiftNetworks shiftNetwork;
  ShiftScheme scheme =
      shiftNetwork.findBestShiftScheme(mapping, vveRandomSeed, vveRandomTries);

  using NodeTy = ArithmeticDagNode<SymbolicValue>;
  using ValueTy = std::shared_ptr<NodeTy>;
  SmallVector<SymbolicValue> inputLeaves(numCiphertexts,
                                         SymbolicValue({ciphertextSize}));
  SmallVector<SmallVector<ValueTy>> groupResults =
      implementRotationGroups(inputLeaves, mapping, scheme, ciphertextSize);

  // The cost is the maximum number of rotations in any group
  Cost maxRotations = 0;
  for (const SmallVector<ValueTy>& groupResult : groupResults) {
    Cost groupRotations = 0;
    // Each group consists of a set of output dag nodes corresponding to each
    // ciphertext in the ciphertext-semantic tensor. The total number of
    // rotations is the sum across all such nodes, noting we're caching common
    // subexpressions.
    for (const ValueTy& v : groupResult) {
      RotationCountVisitor counter;
      groupRotations += v->visit(counter);
    }
    maxRotations = std::max(maxRotations, groupRotations);
  }

  LLVM_DEBUG(llvm::dbgs() << "Estimated cost is " << maxRotations << "\n";);
  return maxRotations;
}

Cost computeCostOfLayoutConversion(int64_t ciphertextSize, Attribute fromLayout,
                                   Attribute toLayout,
                                   std::size_t vveRandomSeed,
                                   unsigned vveRandomTries) {
  if (fromLayout == toLayout) {
    return 0;
  }

  NewLayoutAttr fromLayoutAttr = dyn_cast<NewLayoutAttr>(fromLayout);
  NewLayoutAttr toLayoutAttr = dyn_cast<NewLayoutAttr>(toLayout);

  if (!fromLayoutAttr || !toLayoutAttr) {
    return fromLayout == toLayout ? 0 : 1;
  }

  // Combine random seed with hashes over from- and to-layout, guaranteeing the
  // same result for a given random seed and set of layouts.

  llvm::hash_code fromHash = llvm::hash_value(fromLayoutAttr.getLayout());
  llvm::hash_code toHash = llvm::hash_value(toLayoutAttr.getLayout());
  vveRandomSeed = llvm::hash_combine(vveRandomSeed, fromHash, toHash);

  IntegerRelation rel = fromLayoutAttr.getIntegerRelation();
  std::optional<int64_t> ctUb = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range));
  std::optional<int64_t> ctLb = rel.getConstantBound64(
      BoundType::LB, rel.getVarKindOffset(VarKind::Range));

  if (!ctUb.has_value() || !ctLb.has_value()) {
    llvm::errs() << "Could not determine number of ciphertexts from layout "
                 << fromLayoutAttr << ", assuming cost 1\n";
    return 1;
  }

  int64_t numCiphertexts = ctUb.value() - ctLb.value() + 1;
  return computeCostOfLayoutConversion(numCiphertexts, ciphertextSize,
                                       fromLayoutAttr, toLayoutAttr,
                                       vveRandomSeed, vveRandomTries);
}

}  // namespace heir
}  // namespace mlir
