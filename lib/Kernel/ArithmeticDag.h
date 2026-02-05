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

// An indeterminate variable node, used by internal DAG constructs like
// ForLoopNode to represent the induction variable and iter_arg. These nodes
// must not be set by the user, and instead must have their values set by the
// visitor.
template <typename T>
struct VariableNode {
  std::optional<T> value;
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

// Yield values to a loop.
template <typename T>
struct YieldNode {
  std::vector<std::shared_ptr<ArithmeticDagNode<T>>> elements;
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
struct DivideNode {
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
  std::shared_ptr<ArithmeticDagNode<T>> shift;
};

template <typename T>
struct ExtractNode {
  std::shared_ptr<ArithmeticDagNode<T>> operand;
  std::shared_ptr<ArithmeticDagNode<T>> index;
};

template <typename T>
struct ResultAtNode {
  std::shared_ptr<ArithmeticDagNode<T>> operand;
  size_t index;
};

// A For loop with static bounds
template <typename T>
struct ForLoopNode {
  using NodePtr = std::shared_ptr<ArithmeticDagNode<T>>;
  NodePtr inductionVar;
  std::vector<NodePtr> inits;
  std::vector<NodePtr> iterArgs;
  NodePtr body;
  int32_t lower;
  int32_t upper;
  int32_t step;
};

template <typename T>
struct ArithmeticDagNode {
  using NodePtr = std::shared_ptr<ArithmeticDagNode<T>>;

 public:
  std::variant<ConstantScalarNode, ConstantTensorNode, LeafNode<T>, AddNode<T>,
               SubtractNode<T>, MultiplyNode<T>, DivideNode<T>, PowerNode<T>,
               LeftRotateNode<T>, ExtractNode<T>, VariableNode<T>,
               ForLoopNode<T>, YieldNode<T>, ResultAtNode<T>, SplatNode>
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

  static NodePtr yield(std::vector<NodePtr> values) {
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<YieldNode<T>>(
        YieldNode<T>{std::move(values)});
    return node;
  }

  static NodePtr resultAt(NodePtr value, size_t index) {
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<ResultAtNode<T>>(
        ResultAtNode<T>{std::move(value), index});
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

  static NodePtr div(NodePtr lhs, NodePtr rhs) {
    assert(lhs && rhs && "invalid div");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<DivideNode<T>>(
        DivideNode<T>{std::move(lhs), std::move(rhs)});
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
    // Shift amounts are typically i32 in MLIR
    auto shiftNode =
        constantScalar(static_cast<double>(shift), DagType::integer(32));
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<LeftRotateNode<T>>(
        LeftRotateNode<T>{std::move(tensor), std::move(shiftNode)});
    return node;
  }

  static NodePtr leftRotate(NodePtr tensor, NodePtr shift) {
    assert(tensor && "invalid tensor for leftRotate");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<LeftRotateNode<T>>(
        LeftRotateNode<T>{std::move(tensor), std::move(shift)});
    return node;
  }

  static NodePtr extract(NodePtr tensor, NodePtr index) {
    assert(tensor && index && "invalid tensor or index for extract");
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<ExtractNode<T>>(
        ExtractNode<T>{std::move(tensor), std::move(index)});
    return node;
  }

  static NodePtr extract(NodePtr tensor, size_t index) {
    return extract(
        tensor, constantScalar(static_cast<double>(index), DagType::index()));
  }

  static NodePtr variable() {
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<VariableNode<T>>(
        VariableNode<T>{std::nullopt});
    return node;
  }

  using BodyBuilderFunc =
      std::function<NodePtr(NodePtr inductionVar, const NodePtr& iterArg)>;

  // Construct a loop with a single iter arg. Note that the body builder must
  // have as its root node a YieldNode.
  static NodePtr loop(NodePtr init, int32_t lower, int32_t upper, int32_t step,
                      const BodyBuilderFunc& bodyBuilder = nullptr) {
    assert(init && "invalid init");
    auto inductionVar = variable();
    auto iterArg = variable();
    auto body = bodyBuilder ? bodyBuilder(inductionVar, iterArg) : nullptr;
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<ForLoopNode<T>>(
        ForLoopNode<T>{std::move(inductionVar),
                       {std::move(init)},
                       {std::move(iterArg)},
                       body,
                       lower,
                       upper,
                       step});
    return node;
  }

  using BodyBuilderFuncManyIterArgs = std::function<NodePtr(
      NodePtr inductionVar, const std::vector<NodePtr>& iterArgs)>;

  // Construct a loop with multiple iter args. Note that the body builder must
  // have as its root node a YieldNode.
  static NodePtr loop(
      std::vector<NodePtr> inits, int32_t lower, int32_t upper, int32_t step,
      const BodyBuilderFuncManyIterArgs& bodyBuilder = nullptr) {
    auto inductionVar = variable();
    std::vector<NodePtr> iterArgs(inits.size());
    for (size_t i = 0; i < inits.size(); ++i) {
      iterArgs[i] = variable();
    }

    auto body = bodyBuilder ? bodyBuilder(inductionVar, iterArgs) : nullptr;
    auto node = NodePtr(new ArithmeticDagNode<T>());
    node->node_variant.template emplace<ForLoopNode<T>>(
        ForLoopNode<T>{std::move(inductionVar), std::move(inits),
                       std::move(iterArgs), body, lower, upper, step});
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

    const auto* nodePtr = node.get();
    if (auto it = cache.find(nodePtr); it != cache.end()) {
      return it->second;
    }

    ResultType result = std::visit(*this, node->node_variant);
    cache[nodePtr] = result;
    return result;
  }

  /// An alternate entry point that handles multiple roots.
  std::vector<ResultType> process(
      llvm::ArrayRef<std::shared_ptr<ArithmeticDagNode<T>>> nodes) {
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

  virtual ResultType operator()(const DivideNode<T>& node) {
    assert(false && "Visit logic for DivideNode is not implemented.");
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

  virtual ResultType operator()(const ResultAtNode<T>& node) {
    assert(false && "Visit logic for ResultAtNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const VariableNode<T>& node) {
    assert(false && "Visit logic for VariableNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const ForLoopNode<T>& node) {
    assert(false && "Visit logic for ForLoopNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const YieldNode<T>& node) {
    assert(false && "Visit logic for YieldNode is not implemented.");
    return ResultType();
  }

  virtual ResultType operator()(const SplatNode& node) {
    assert(false && "Visit logic for SplatNode is not implemented.");
    return ResultType();
  }

 protected:
  void clearCache() { cache.clear(); }

  void clearCacheEntry(const ArithmeticDagNode<T>* node) { cache.erase(node); }

  void clearSubtreeCache(const std::shared_ptr<ArithmeticDagNode<T>>& node) {
    if (!node) return;

    clearCacheEntry(node.get());

    // Recursively clear cache for child nodes
    std::visit(
        [this](auto&& n) {
          using NodeType = std::decay_t<decltype(n)>;
          if constexpr (std::is_same_v<NodeType, AddNode<T>> ||
                        std::is_same_v<NodeType, SubtractNode<T>> ||
                        std::is_same_v<NodeType, MultiplyNode<T>> ||
                        std::is_same_v<NodeType, DivideNode<T>>) {
            clearSubtreeCache(n.left);
            clearSubtreeCache(n.right);
          } else if constexpr (std::is_same_v<NodeType, PowerNode<T>>) {
            clearSubtreeCache(n.base);
          } else if constexpr (std::is_same_v<NodeType, LeftRotateNode<T>>) {
            clearSubtreeCache(n.operand);
            clearSubtreeCache(n.shift);
          } else if constexpr (std::is_same_v<NodeType, ExtractNode<T>>) {
            clearSubtreeCache(n.operand);
            clearSubtreeCache(n.index);
          } else if constexpr (std::is_same_v<NodeType, ResultAtNode<T>>) {
            clearSubtreeCache(n.operand);
          } else if constexpr (std::is_same_v<NodeType, ForLoopNode<T>>) {
            clearSubtreeCache(n.inductionVar);
            for (const auto& init : n.inits) {
              clearSubtreeCache(init);
            }
            for (const auto& iterArg : n.iterArgs) {
              clearSubtreeCache(iterArg);
            }
            clearSubtreeCache(n.body);
          } else if constexpr (std::is_same_v<NodeType, YieldNode<T>>) {
            for (const auto& element : n.elements) {
              clearSubtreeCache(element);
            }
          }
        },
        node->node_variant);
  }

 private:
  std::unordered_map<const ArithmeticDagNode<T>*, ResultType> cache;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ARITHMETICDAG_H_
