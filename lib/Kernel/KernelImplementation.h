#ifndef LIB_KERNEL_KERNELIMPLEMENTATION_H_
#define LIB_KERNEL_KERNELIMPLEMENTATION_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

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

// Returns an arithmetic DAG that implements a matvec kernel. Ensure this is
// only generated for T a subclass of AbstractValue.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMatvec(KernelName kernelName, const T& matrix, const T& vector) {
  using NodeTy = ArithmeticDagNode<T>;
  assert(kernelName == KernelName::MatvecDiagonal);
  auto matrixDag = NodeTy::leaf(matrix);
  auto vectorDag = NodeTy::leaf(vector);

  int numRows = matrix.getShape()[0];
  assert(numRows > 0);

  auto firstTerm = NodeTy::mul(NodeTy::leftRotate(vectorDag, 0),
                               NodeTy::extract(matrixDag, 0));

  auto accumulatedSum = firstTerm;
  for (int i = 1; i < numRows; ++i) {
    auto term = NodeTy::mul(NodeTy::leftRotate(vectorDag, i),
                            NodeTy::extract(matrixDag, i));
    accumulatedSum = NodeTy::add(accumulatedSum, term);
  }
  return accumulatedSum;
}

// Returns an arithmetic DAG that implements a rotate and reduce op. Ensure
// this is only generated for T a subclass of AbstractValue.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduce(const T& vector, std::optional<T> plaintexts,
                         int64_t period, int64_t steps) {
  using NodeTy = ArithmeticDagNode<T>;
  auto vectorDag = NodeTy::leaf(vector);

  if (!plaintexts.has_value()) {
    for (int64_t shiftSize = steps / 2; shiftSize > 0; shiftSize /= 2) {
      auto rotated = NodeTy::leftRotate(vectorDag, shiftSize * period);
      auto added = NodeTy::add(vectorDag, rotated);
      vectorDag = added;
    }
    return vectorDag;
  }

  auto plaintextsDag = NodeTy::leaf(*plaintexts);

  // Use a value of sqrt(n) as the baby step / giant step size.
  int64_t numBabySteps = static_cast<int64_t>(std::floor(std::sqrt(steps)));
  if (steps % numBabySteps != 0) {
    // Find the nearest divisible number to use for baby step
    // TODO(#2162): determine the right tradeoff here
    int lower = numBabySteps;
    int upper = numBabySteps;

    while (steps % lower != 0 && steps % upper != steps) {
      lower--;
      upper++;
    }

    if (steps % lower == 0 && lower > 1) {
      numBabySteps = lower;
    } else if (steps % upper == 0) {
      numBabySteps = upper;
    } else {
      numBabySteps = steps;
    }
  }

  int64_t giantStepSize = numBabySteps;
  int64_t numGiantSteps = steps / numBabySteps;

  // Compute sqrt(n) ciphertext rotations of the input as baby-steps.
  SmallVector<std::shared_ptr<NodeTy>> babyStepVals;
  babyStepVals.push_back(vectorDag);  // rot by zero
  for (int64_t i = 1; i < numBabySteps; ++i) {
    babyStepVals.push_back(NodeTy::leftRotate(vectorDag, period * i));
  }

  // Compute the inner baby step sums.
  std::shared_ptr<NodeTy> result = nullptr;
  for (int64_t j = 0; j < numGiantSteps; ++j) {
    std::shared_ptr<NodeTy> innerSum = nullptr;
    // The rotation used for the plaintext
    int64_t plaintextRotationAmount = -giantStepSize * j * period;
    for (int64_t i = 0; i < numBabySteps; ++i) {
      size_t extractionIndex = i + j * giantStepSize;
      auto plaintext = NodeTy::extract(plaintextsDag, extractionIndex);
      auto rotatedPlaintext =
          NodeTy::leftRotate(plaintext, plaintextRotationAmount);
      auto multiplied = NodeTy::mul(rotatedPlaintext, babyStepVals[i]);
      innerSum =
          innerSum == nullptr ? multiplied : NodeTy::add(innerSum, multiplied);
    }

    auto rotatedSum = NodeTy::leftRotate(innerSum, period * j * giantStepSize);
    result = result == nullptr ? rotatedSum : NodeTy::add(result, rotatedSum);
  }

  return result;
}

// Returns an arithmetic DAG that implements the Halevi-Shoup matrix
// multiplication algorithm. This implementation uses a rotate-and-reduce
// operation, followed by a summation of partial sums if the matrix is not
// square.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementHaleviShoup(const T& vector, const T& matrix,
                     std::vector<int64_t> originalMatrixShape) {
  using NodeTy = ArithmeticDagNode<T>;
  int64_t numRotations = matrix.getShape()[0];

  auto rotateAndReduceResult = implementRotateAndReduce<T>(
      vector, std::optional<T>(matrix), /*period=*/1,
      /*steps=*/numRotations);

  auto summedShifts = rotateAndReduceResult;

  int64_t matrixNumRows = nextPowerOfTwo(originalMatrixShape[0]);
  int64_t matrixNumCols = nextPowerOfTwo(originalMatrixShape[1]);

  if (matrixNumRows == matrixNumCols) {
    return summedShifts;
  }

  // Post-processing partial-rotate-and-reduce step required for
  // squat-diagonal packing.
  int64_t numShifts = (int64_t)(log2(matrixNumCols) - log2(matrixNumRows));
  int64_t shift = matrixNumCols / 2;
  for (int64_t i = 0; i < numShifts; ++i) {
    auto rotated = NodeTy::leftRotate(summedShifts, shift);
    summedShifts = NodeTy::add(summedShifts, rotated);
    shift /= 2;
  }

  return summedShifts;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_KERNELIMPLEMENTATION_H_
