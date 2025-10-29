#ifndef LIB_KERNEL_KERNELIMPLEMENTATION_H_
#define LIB_KERNEL_KERNELIMPLEMENTATION_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

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
                         int64_t period, int64_t steps,
                         const std::string& reduceOp = "arith.addi") {
  using NodeTy = ArithmeticDagNode<T>;
  auto vectorDag = NodeTy::leaf(vector);

  auto performReduction = [&](std::shared_ptr<NodeTy> left,
                              std::shared_ptr<NodeTy> right) {
    if (reduceOp == "arith.addi" || reduceOp == "arith.addf") {
      return NodeTy::add(left, right);
    }

    if (reduceOp == "arith.muli" || reduceOp == "arith.mulf") {
      return NodeTy::mul(left, right);
    }

    // Default to add for unknown operations
    return NodeTy::add(left, right);
  };

  if (!plaintexts.has_value()) {
    for (int64_t shiftSize = steps / 2; shiftSize > 0; shiftSize /= 2) {
      auto rotated = NodeTy::leftRotate(vectorDag, shiftSize * period);
      auto reduced = performReduction(vectorDag, rotated);
      vectorDag = reduced;
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
      innerSum = innerSum == nullptr ? multiplied
                                     : performReduction(innerSum, multiplied);
    }

    auto rotatedSum = NodeTy::leftRotate(innerSum, period * j * giantStepSize);
    result =
        result == nullptr ? rotatedSum : performReduction(result, rotatedSum);
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

// Returns an arithmetic DAG that implements the bicyclic matrix multiplication
// algorithm.
//
// The input matrices packedA and packedB are assumed to be properly packed to
// meet the conditions for bicyclic multiplication. That is: both matrices are
// zero-padded so that their dimensions are coprime, they are cyclically
// repeated to fill all the slots of the ciphertext, and they are packed
// according to the bicyclic ordering.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBicyclicMatmul(const T& packedA, const T& packedB, int64_t m,
                        int64_t n, int64_t p) {
  using NodeTy = ArithmeticDagNode<T>;
  auto packedADag = NodeTy::leaf(packedA);
  auto packedBDag = NodeTy::leaf(packedB);

  // This implements the BMM-I algorithm from https://eprint.iacr.org/2024/1762
  // with a simplification of the rotation formula in Sec 5.2.1 (equation 9) of
  // https://eprint.iacr.org/2025/1200
  //
  // C = sum_{c=0}^{n-1} rot(A, r1(c)) * rot(B, r2(c))
  //
  // where
  //
  //  r1(c) = cm(m^{-1} mod n) mod mn
  //  r2(c) = cp(p^{-1} mod n) mod np
  //
  APInt mAPInt = APInt(64, m);
  APInt nAPInt = APInt(64, n);
  APInt pAPInt = APInt(64, p);

  APInt mInvModN = multiplicativeInverse(mAPInt, nAPInt);
  APInt pInvModN = multiplicativeInverse(pAPInt, nAPInt);

  // The part of r1(c), r2(c) that is independent of the loop iterations
  int64_t r1Const = m * mInvModN.getSExtValue();
  int64_t r2Const = p * pInvModN.getSExtValue();

  std::shared_ptr<NodeTy> result = nullptr;
  for (int i = 0; i < n; ++i) {
    int64_t shiftA = (i * r1Const) % (m * n);
    int64_t shiftB = (i * r2Const) % (n * p);
    auto rotA = NodeTy::leftRotate(packedADag, shiftA);
    auto rotB = NodeTy::leftRotate(packedBDag, shiftB);
    auto term = NodeTy::mul(rotA, rotB);
    result = result == nullptr ? term : NodeTy::add(result, term);
  }
  return result;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_KERNELIMPLEMENTATION_H_
