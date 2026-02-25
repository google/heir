#ifndef LIB_UTILS_ARITHMETICDAG_H_
#define LIB_UTILS_ARITHMETICDAG_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// Type system for ArithmeticDag nodes
struct IntegerType {
  unsigned bitWidth;
};

struct FloatType {
  unsigned bitWidth;
};

struct IndexType {};

struct IntTensorType {
  unsigned bitWidth;
  std::vector<int64_t> shape;
};

struct FloatTensorType {
  unsigned bitWidth;
  std::vector<int64_t> shape;
};

struct DagType {
  std::variant<IntegerType, FloatType, IndexType, IntTensorType,
               FloatTensorType>
      type_variant;

  static DagType integer(unsigned bitWidth) {
    DagType t;
    t.type_variant = IntegerType{bitWidth};
    return t;
  }

  static DagType floatTy(unsigned bitWidth) {
    DagType t;
    t.type_variant = FloatType{bitWidth};
    return t;
  }

  static DagType index() {
    DagType t;
    t.type_variant = IndexType{};
    return t;
  }

  static DagType intTensor(unsigned bitWidth, std::vector<int64_t> shape) {
    DagType t;
    t.type_variant = IntTensorType{bitWidth, std::move(shape)};
    return t;
  }

  static DagType floatTensor(unsigned bitWidth, std::vector<int64_t> shape) {
    DagType t;
    t.type_variant = FloatTensorType{bitWidth, std::move(shape)};
    return t;
  }
};

// This file contains a generic DAG structure that can be used for representing
// arithmetic DAGs with leaf nodes of various types.
template <typename T>
struct ArithmeticDagNode;

// A leaf node for the DAG
template <typename T>
struct LeafNode {
  T value;
};

struct ConstantScalarNode {
  double value;
  DagType type;
};

struct ConstantTensorNode {
  std::vector<double> value;
  DagType type;
};

struct SplatNode {
  double value;
  DagType type;
};

template <typename T>
struct AddNode {
  std::shared_ptr<ArithmeticDagNode<T>> left;
  std::shared_ptr<ArithmeticDagNode<T>> right;
};

template <typename T>
struct SubtractNode {
  std::shared_ptr<ArithmeticDagNode<T>> left;
  std::shared_ptr<ArithmeticDagNode<T>> right;
};

template <typename T>
struct MultiplyNode {
  std::shared_ptr<ArithmeticDagNode<T>> left;
  std::shared_ptr<ArithmeticDagNode<T>> right;
};

template <typename T>
struct FloorDivNode {
  std::shared_ptr<ArithmeticDagNode<T>> left;
  int divisor;
};

template <typename T>
struct PowerNode {
  std::shared_ptr<ArithmeticDagNode<T>> base;
  size_t exponent;
};

template <typename T>
struct LeftRotateNode {
  std::shared_ptr<ArithmeticDagNode<T>> operand;
  std::shared_ptr<ArithmeticDagNode<T>> shift;
};

template <typename T>
struct ExtractNode {
  std::shared_ptr<ArithmeticDagNode<T>> operand;
  size_t index;
};

template <typename T>
struct ArithmeticDagNode {
  using NodePtr = std::shared_ptr<ArithmeticDagNode<T>>;

 public:
  std::variant<ConstantScalarNode, ConstantTensorNode, LeafNode<T>, AddNode<T>,
               SubtractNode<T>, MultiplyNode<T>, FloorDivNode<T>, PowerNode<T>,
               LeftRotateNode<T>, ExtractNode<T>, SplatNode>
      node_variant;

  explicit ArithmeticDagNode(const T& value)
      : node_variant(LeafNode<T>{value}) {}
  explicit ArithmeticDagNode(T&& value)
      : node_variant(LeafNode<T>{std::move(value)}) {}

 private:
  ArithmeticDagNode() = default;

 public:
  // Static factory methods
  static NodePtr leaf(const T& value) {
    // This factory method differs from the others because T may not have a
    // default constructor to use with emplace. In that case, we need to rely
    // on the move or copy constructors, which corresponds to the two
    // ArithmeticDagNode constructors above.
    return NodePtr(new ArithmeticDagNode<T>(value));
  }

  static NodePtr constantScalar(double constant, DagType type) {
    auto node = NodePtr(new ArithmeticDagNode<T>());
    // Note, to satisfy variant we need to use aggregate initialization inside
    // emplace
    node->node_variant.template emplace<ConstantScalarNode>(
        ConstantScalarNode{constant, std::move(type)});
    return node;
  }

  static NodePtr constantTensor(std::vector<double> constant, DagType type) {
    auto node = NodePtr(new ArithmeticDagNode<T>());
    // Note, to satisfy variant we need to use aggregate initialization inside
    // emplace
    node->node_variant.template emplace<ConstantTensorNode>(
        ConstantTensorNode{std::move(constant), std::move(type)});
    return node;
  }

  static NodePtr splat(double constant, DagType type) {
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<SplatNode>(
        SplatNode{constant, std::move(type)});
    return node;
  }

  static NodePtr add(NodePtr lhs, NodePtr rhs) {
    assert(lhs && rhs && "invalid add");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<AddNode<T>>(
        AddNode<T>{std::move(lhs), std::move(rhs)});
    return node;
  }

  static NodePtr sub(NodePtr lhs, NodePtr rhs) {
    assert(lhs && rhs && "invalid sub");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<SubtractNode<T>>(
        SubtractNode<T>{std::move(lhs), std::move(rhs)});
    return node;
  }

  static NodePtr mul(NodePtr lhs, NodePtr rhs) {
    assert(lhs && rhs && "invalid mul");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<MultiplyNode<T>>(
        MultiplyNode<T>{std::move(lhs), std::move(rhs)});
    return node;
  }

  static NodePtr floorDiv(NodePtr lhs, int rhs) {
    assert(lhs && "invalid div");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<FloorDivNode<T>>(
        FloorDivNode<T>{std::move(lhs), rhs});
    return node;
  }

  static NodePtr power(NodePtr base, size_t exponent) {
    assert(base && "invalid base for power");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<PowerNode<T>>(
        PowerNode<T>{std::move(base), exponent});
    return node;
  }

  static NodePtr leftRotate(NodePtr tensor, int64_t shift) {
    assert(tensor && "invalid tensor for leftRotate");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    auto constantShift = constantScalar(shift, DagType::index());
    node->node_variant.template emplace<LeftRotateNode<T>>(
        LeftRotateNode<T>{std::move(tensor), std::move(constantShift)});
    return node;
  }

  static NodePtr leftRotate(NodePtr tensor, NodePtr shift) {
    assert(tensor && "invalid tensor for leftRotate");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<LeftRotateNode<T>>(
        LeftRotateNode<T>{std::move(tensor), std::move(shift)});
    return node;
  }

  static NodePtr extract(NodePtr tensor, size_t index) {
    assert(tensor && "invalid tensor for extract");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<ExtractNode<T>>(
        ExtractNode<T>{std::move(tensor), index});
    return node;
  }

  ArithmeticDagNode(const ArithmeticDagNode&) = default;
  ArithmeticDagNode& operator=(const ArithmeticDagNode&) = default;
  ArithmeticDagNode(ArithmeticDagNode&&) noexcept = default;
  ArithmeticDagNode& operator=(ArithmeticDagNode&&) noexcept = default;

  // Visitor pattern
  template <typename VisitorFunc>
  decltype(auto) visit(VisitorFunc&& visitor) {
    return std::visit(std::forward<VisitorFunc>(visitor), node_variant);
  }

  template <typename VisitorFunc>
  decltype(auto) visit(VisitorFunc&& visitor) const {
    return std::visit(std::forward<VisitorFunc>(visitor), node_variant);
  }
};

/// A base class for visitors that caches intermediate results.
///
/// Template parameters:
///   T: The type of the leaf nodes.
///   ResultType: The type of the result of the visit.
template <typename T, typename ResultType>
class CachingVisitor {
  using NodePtr = std::shared_ptr<ArithmeticDagNode<T>>;

 public:
  virtual ~CachingVisitor() = default;

  /// The main entry point that contains the caching logic.
  ResultType process(const NodePtr& node) {
    assert(node != nullptr && "invalid null node!");

    const auto* nodePtr = node.get();
    if (auto it = cache.find(nodePtr); it != cache.end()) {
      return it->second;
    }

    ResultType result = std::visit(*this, node->node_variant);
    cache[nodePtr] = result;
    return result;
  }

  /// An alternate entry point that handles multiple roots.
  std::vector<ResultType> process(llvm::ArrayRef<NodePtr> nodes) {
    std::vector<ResultType> results;
    results.reserve(nodes.size());
    for (const auto& node : nodes) {
      results.push_back(process(node));
    }
    return results;
  }

  // --- Virtual Visit Methods ---
  // Derived classes must override these for the node types they support.
  //
  // If some implementations are omitted, the derived class must add
  //
  //   using CachingVisitor<double, double>::operator();
  //
  // to avoid the name-hiding that occurs when a derived class overrides
  // one of the operator() methods.

  virtual ResultType operator()(const ConstantScalarNode& node) {
    assert(false && "Visit logic for ConstantScalarNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const ConstantTensorNode& node) {
    assert(false && "Visit logic for ConstantTensorNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const LeafNode<T>& node) {
    assert(false && "Visit logic for LeafNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const AddNode<T>& node) {
    assert(false && "Visit logic for AddNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const SubtractNode<T>& node) {
    assert(false && "Visit logic for SubtractNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const MultiplyNode<T>& node) {
    assert(false && "Visit logic for MultiplyNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const FloorDivNode<T>& node) {
    assert(false && "Visit logic for FloorDivNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const PowerNode<T>& node) {
    assert(false && "Visit logic for PowerNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const LeftRotateNode<T>& node) {
    assert(false && "Visit logic for LeftRotateNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const ExtractNode<T>& node) {
    assert(false && "Visit logic for ExtractNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const SplatNode& node) {
    assert(false && "Visit logic for SplatNode is not implemented.");
    return ResultType();
  }

 private:
  std::unordered_map<const ArithmeticDagNode<T>*, ResultType> cache;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ARITHMETICDAG_H_
