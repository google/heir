#include "lib/Transforms/LayoutOptimization/LayoutConversionCost.h"

#include <cstdint>

#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

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
using presburger::IntegerRelation;
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

int64_t computeCostOfLayoutConversion(int64_t numCiphertexts,
                                      int64_t ciphertextSize,
                                      NewLayoutAttr fromLayout,
                                      NewLayoutAttr toLayout) {
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
  ShiftScheme scheme = shiftNetwork.findShiftScheme(mapping);

  using NodeTy = ArithmeticDagNode<SymbolicValue>;
  using ValueTy = std::shared_ptr<NodeTy>;
  SmallVector<SymbolicValue> inputLeaves(numCiphertexts,
                                         SymbolicValue({ciphertextSize}));
  SmallVector<SmallVector<ValueTy>> groupResults =
      implementRotationGroups(inputLeaves, mapping, scheme, ciphertextSize);

  // The cost is the maximum number of rotations in any group
  int64_t maxRotations = 0;
  for (const SmallVector<ValueTy>& groupResult : groupResults) {
    int64_t groupRotations = 0;
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

}  // namespace heir
}  // namespace mlir
