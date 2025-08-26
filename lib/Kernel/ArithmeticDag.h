#ifndef LIB_UTILS_ARITHMETICDAG_H_
#define LIB_UTILS_ARITHMETICDAG_H_

#include <cassert>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>

namespace mlir {
namespace heir {
namespace kernel {

// This file contains a generic DAG structure that can be used for representing
// arithmetic DAGs with leaf nodes of various types.
template <typename T>
struct ArithmeticDagNode;

// A leaf node for the DAG
template <typename T>
struct LeafNode {
  T value;
};

struct ConstantNode {
  double value;
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
struct PowerNode {
  std::shared_ptr<ArithmeticDagNode<T>> base;
  size_t exponent;
};

template <typename T>
struct LeftRotateNode {
  std::shared_ptr<ArithmeticDagNode<T>> operand;
  int64_t shift;
};

template <typename T>
struct ExtractNode {
  std::shared_ptr<ArithmeticDagNode<T>> operand;
  size_t index;
};

template <typename T>
struct ArithmeticDagNode {
 public:
  std::variant<ConstantNode, LeafNode<T>, AddNode<T>, SubtractNode<T>,
               MultiplyNode<T>, PowerNode<T>, LeftRotateNode<T>, ExtractNode<T>>
      node_variant;

  explicit ArithmeticDagNode(const T& value)
      : node_variant(LeafNode<T>{value}) {}
  explicit ArithmeticDagNode(T&& value)
      : node_variant(LeafNode<T>{std::move(value)}) {}

 private:
  ArithmeticDagNode() = default;

 public:
  // Static factory methods
  static std::shared_ptr<ArithmeticDagNode<T>> leaf(const T& value) {
    // This factory method differs from the others because T may not have a
    // default constructor to use with emplace. In that case, we need to rely
    // on the move or copy constructors, which corresponds to the two
    // ArithmeticDagNode constructors above.
    return std::shared_ptr<ArithmeticDagNode<T>>(
        new ArithmeticDagNode<T>(value));
  }

  static std::shared_ptr<ArithmeticDagNode<T>> constant(double constant) {
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
    // Note, to satisfy variant we need to use aggregate initialization inside
    // emplace
    node->node_variant.template emplace<ConstantNode>(ConstantNode{constant});
    return node;
  }

  static std::shared_ptr<ArithmeticDagNode<T>> add(
      std::shared_ptr<ArithmeticDagNode<T>> lhs,
      std::shared_ptr<ArithmeticDagNode<T>> rhs) {
    assert(lhs && rhs && "invalid add");
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<AddNode<T>>(
        AddNode<T>{std::move(lhs), std::move(rhs)});
    return node;
  }

  static std::shared_ptr<ArithmeticDagNode<T>> sub(
      std::shared_ptr<ArithmeticDagNode<T>> lhs,
      std::shared_ptr<ArithmeticDagNode<T>> rhs) {
    assert(lhs && rhs && "invalid sub");
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<SubtractNode<T>>(
        SubtractNode<T>{std::move(lhs), std::move(rhs)});
    return node;
  }

  static std::shared_ptr<ArithmeticDagNode<T>> mul(
      std::shared_ptr<ArithmeticDagNode<T>> lhs,
      std::shared_ptr<ArithmeticDagNode<T>> rhs) {
    assert(lhs && rhs && "invalid mul");
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<MultiplyNode<T>>(
        MultiplyNode<T>{std::move(lhs), std::move(rhs)});
    return node;
  }

  static std::shared_ptr<ArithmeticDagNode<T>> power(
      std::shared_ptr<ArithmeticDagNode<T>> base, size_t exponent) {
    assert(base && "invalid base for power");
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<PowerNode<T>>(
        PowerNode<T>{std::move(base), exponent});
    return node;
  }

  static std::shared_ptr<ArithmeticDagNode<T>> leftRotate(
      std::shared_ptr<ArithmeticDagNode<T>> tensor, int64_t shift) {
    assert(tensor && "invalid tensor for leftRotate");
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<LeftRotateNode<T>>(
        LeftRotateNode<T>{std::move(tensor), shift});
    return node;
  }

  static std::shared_ptr<ArithmeticDagNode<T>> extract(
      std::shared_ptr<ArithmeticDagNode<T>> tensor, size_t index) {
    assert(tensor && "invalid tensor for extract");
    auto node =
        std::shared_ptr<ArithmeticDagNode<T>>(new ArithmeticDagNode<T>());
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
 public:
  virtual ~CachingVisitor() = default;

  /// The main entry point that contains the caching logic.
  ResultType process(const std::shared_ptr<ArithmeticDagNode<T>>& node) {
    assert(node != nullptr && "invalid null node!");

    const auto* node_ptr = node.get();
    if (auto it = cache.find(node_ptr); it != cache.end()) {
      return it->second;
    }

    ResultType result = std::visit(*this, node->node_variant);
    cache[node_ptr] = result;
    return result;
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

  virtual ResultType operator()(const ConstantNode& node) {
    assert(false && "Visit logic for ConstantNode is not implemented.");
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

 private:
  std::unordered_map<const ArithmeticDagNode<T>*, ResultType> cache;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ARITHMETICDAG_H_
