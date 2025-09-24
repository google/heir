#ifndef LIB_KERNEL_ABSTRACTVALUE_H_
#define LIB_KERNEL_ABSTRACTVALUE_H_

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// Kernel implementations are designed to work in two regimes:
//
// - Testing with literal values as leaf nodes (i.e., std::vector<int>). The
//   resulting DAG can be evaluated directly using EvalVisitor.
//
// - Using SSA values as leaf nodes (::mlir::Value), the resulting DAG can be
//   converted to MLIR using IRMaterializingVisitor.
//
// To keep a single implementation of the kernel generator and visitors, we
// define an AbstractValue interface that exposes the minimal needed
// functionality of the kernel generation code, and the two regimes may use
// different concrete types for testing (LiteralValue) and IR generation
// (SSAValue), which are thin wrappers around the actual data.
class AbstractValue {
 public:
  virtual ~AbstractValue() = default;

  // Returns a vector of the size of each tensor dimension if the value is a
  // tensor. If a scalar, returns empty vector.
  virtual std::vector<int64_t> getShape() const = 0;
};

// A type that holds a literal double.
class LiteralDouble : public AbstractValue {
 public:
  LiteralDouble() : d(0.0) {}
  LiteralDouble(double d) : d(d) {}

  double getValue() const { return d; }

  std::vector<int64_t> getShape() const override { return {}; }

 private:
  double d;
};

// A type that holds a literal tensor, which can either be a 1D or 2D tensor.
//
// More variants must be added to support higher-dimensional input/output
// tensors.
class LiteralValue : public AbstractValue {
  using CiphertextSemanticTensor =
      std::variant<std::vector<int>, std::vector<std::vector<int>>>;

 public:
  LiteralValue() : tensor({}) {}
  LiteralValue(const CiphertextSemanticTensor& tensor) : tensor(tensor) {}
  LiteralValue(std::vector<int> vec) : tensor(vec) {}
  LiteralValue(std::vector<std::vector<int>> vec) : tensor(vec) {}

  const CiphertextSemanticTensor& getTensor() const { return tensor; }

  std::vector<int64_t> getShape() const override {
    return std::visit(
        [](auto&& arg) -> std::vector<int64_t> {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::vector<int>>) {
            return {static_cast<int64_t>(arg.size())};
          } else if constexpr (std::is_same_v<T,
                                              std::vector<std::vector<int>>>) {
            if (arg.empty()) return {0, 0};
            return {static_cast<int64_t>(arg.size()),
                    static_cast<int64_t>(arg[0].size())};
          } else {
            assert(false && "Unsupported tensor type");
            return {};
          }
        },
        tensor);
  }

 private:
  CiphertextSemanticTensor tensor;
};

class SSAValue : public AbstractValue {
 public:
  SSAValue(::mlir::Value value) : value(value) {}
  ::mlir::Value getValue() const { return value; }

  std::vector<int64_t> getShape() const override {
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
      return tensorType.getShape();
    }
    return {};
  }

 private:
  ::mlir::Value value;
};

/// An AbstractValue with no backing data, used for analysis.
class SymbolicValue : public AbstractValue {
 public:
  SymbolicValue(const std::vector<int64_t>& shape) : shape(shape) {}
  std::vector<int64_t> getShape() const override { return shape; }

 private:
  std::vector<int64_t> shape;
};

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_ABSTRACTVALUE_H_
