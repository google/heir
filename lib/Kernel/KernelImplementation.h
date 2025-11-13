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

// A function that generalizes the reduction operation in all kernels in this
// file. E.g., whether to use `add` or `mul`
template <typename T>
using DagReducer = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    std::shared_ptr<ArithmeticDagNode<T>>,
    std::shared_ptr<ArithmeticDagNode<T>>)>;

template <typename T>
using DagExtractor = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    std::shared_ptr<ArithmeticDagNode<T>>, int64_t)>;

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

// Returns an arithmetic DAG that implements a logarithmic rotate-and-reduce
// accumulation of an input ciphertext.
//
// This is a special case of `tensor_ext.rotate_and_reduce`
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduceAccumulation(const T& vector, int64_t period,
                                     int64_t steps, DagReducer<T> reduceFunc) {
  using NodeTy = ArithmeticDagNode<T>;
  auto vectorDag = NodeTy::leaf(vector);

  for (int64_t shiftSize = steps / 2; shiftSize > 0; shiftSize /= 2) {
    auto rotated = NodeTy::leftRotate(vectorDag, shiftSize * period);
    auto reduced = reduceFunc(vectorDag, rotated);
    vectorDag = reduced;
  }
  return vectorDag;
}

// A function that generalizes the choice of rotation for the "baby stepped
// operand" of a baby-step giant-step algorithm. This is required because
// the rotation used in Halevi-Shoup matvec differs from that of bicyclic
// matmul.
using DerivedRotationIndexFn = std::function<int64_t(
    // giant step size
    int64_t,
    // current giant step index
    int64_t,
    // current baby step index
    int64_t,
    // period
    int64_t)>;

inline int64_t defaultDerivedRotationIndexFn(int64_t giantStepSize,
                                             int64_t giantStepIndex,
                                             int64_t babyStepIndex,
                                             int64_t period) {
  return -giantStepSize * giantStepIndex * period;
}

// Returns an arithmetic DAG that implements a baby-step-giant-step
// rotate-and-reduce accumulation between an input ciphertext
// (giantSteppedOperand) and an abstraction over the other argument
// (babySteppedOperand). In particular, the babySteppedOperand may be a list of
// plaintexts like in Halevi-Shoup matvec, or a single ciphertext like in
// bicyclic matmul, and this abstracts over both by taking in an extraction
// callback.
//
// This is a special case of `tensor_ext.rotate_and_reduce`, but with the added
// abstractions it also supports situations not currently expressible by
// `tensor_ext.rotate_and_reduce`.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBabyStepGiantStep(
    const T& giantSteppedOperand, const T& babySteppedOperand, int64_t period,
    int64_t steps, DagExtractor<T> extractFunc,
    const DerivedRotationIndexFn& derivedRotationIndexFn =
        defaultDerivedRotationIndexFn) {
  using NodeTy = ArithmeticDagNode<T>;
  auto giantSteppedDag = NodeTy::leaf(giantSteppedOperand);
  auto babySteppedDag = NodeTy::leaf(babySteppedOperand);

  // Use a value of sqrt(n) as the baby step / giant step size.
  int64_t numBabySteps = static_cast<int64_t>(std::ceil(std::sqrt(steps)));
  int64_t giantStepSize = numBabySteps;
  // numGiantSteps = ceil(steps / numBabySteps)
  int64_t numGiantSteps = (steps + numBabySteps - 1) / numBabySteps;

  // Compute sqrt(n) ciphertext rotations of the input as baby-steps.
  SmallVector<std::shared_ptr<NodeTy>> babyStepVals;
  babyStepVals.push_back(giantSteppedDag);  // rot by zero
  for (int64_t i = 1; i < numBabySteps; ++i) {
    babyStepVals.push_back(NodeTy::leftRotate(giantSteppedDag, period * i));
  }

  // Compute the inner baby step sums.
  std::shared_ptr<NodeTy> result = nullptr;
  for (int64_t j = 0; j < numGiantSteps; ++j) {
    std::shared_ptr<NodeTy> innerSum = nullptr;
    for (int64_t i = 0; i < numBabySteps; ++i) {
      if (j * giantStepSize + i >= steps) {
        break;
      }
      int64_t innerRotAmount =
          derivedRotationIndexFn(giantStepSize, j, i, period);
      size_t extractionIndex = i + j * giantStepSize;
      auto plaintext = extractFunc(babySteppedDag, extractionIndex);
      auto rotatedPlaintext = NodeTy::leftRotate(plaintext, innerRotAmount);
      auto multiplied = NodeTy::mul(rotatedPlaintext, babyStepVals[i]);
      innerSum =
          innerSum == nullptr ? multiplied : NodeTy::add(innerSum, multiplied);
    }

    auto rotatedSum = NodeTy::leftRotate(innerSum, period * j * giantStepSize);
    result = result == nullptr ? rotatedSum : NodeTy::add(result, rotatedSum);
  }

  return result;
}

// Returns an arithmetic DAG that implements a tensor_ext.rotate_and_reduce op.
//
// See TensorExtOps.td docs for RotateAndReduceOp for more details.
//
// The `vector` argument is a ciphertext value that will be rotated O(sqrt(n))
// times when the `plaintexts` argument is set (Baby Step Giant Step), or
// O(log(n)) times when the `plaintexts` argument is not set (log-style
// rotate-and-reduce accumulation).
//
// The `plaintexts` argument, when present, represents a vector of pre-packed
// plaintexts that will be rotated and multiplied with the rotated `vector`
// argument in BSGS style.
//
// Note that using this kernel results in places in the pipeline where a
// plaintext type is rotated, but most FHE implementations don't have a
// plaintext rotation operation (it would be wasteful) and instead expect the
// "plaintext rotation" to apply to the cleartext. HEIR has places in the
// pipeline that support this by converting a rotate(encode(cleartext)) to
// encode(rotate(cleartext)).
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduce(const T& vector, std::optional<T> plaintexts,
                         int64_t period, int64_t steps,
                         const std::string& reduceOp = "arith.addi") {
  using NodeTy = ArithmeticDagNode<T>;
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
    return implementRotateAndReduceAccumulation<T>(vector, period, steps,
                                                   performReduction);
  }

  assert(reduceOp == "arith.addi" ||
         reduceOp == "arith.addf" &&
             "Baby-step-giant-step rotate-and-reduce only supports addition "
             "as the reduction operation");

  auto extractFunc = [](std::shared_ptr<NodeTy> babySteppedDag,
                        int64_t extractionIndex) {
    return NodeTy::extract(babySteppedDag, extractionIndex);
  };

  return implementBabyStepGiantStep<T>(vector, plaintexts.value(), period,
                                       steps, extractFunc);
}

// Returns an arithmetic DAG that implements a baby-step-giant-step between
// ciphertexts.
//
// This implements equation 21 in 6.2.2 of LKAA25: "Tricycle: Private
// Transformer Inference with Tricyclic Encodings"
// https://eprint.iacr.org/2025/1200
//
// This differs from the above implementRotateAndReduce in that, instead of a
// set of pre-computed plaintexts, both arguments are individual ciphertexts.
// Normally with one ciphertext, the naive approach uses n - 1 rotations that
// BSGS reduces to c sqrt(n) + O(1) rotations, if both inputs are ciphertexts
// then it converts 2n - 2 total rotations to n + c sqrt(n) + O(1) rotations.
// Essentially, the "n to sqrt(n)" redution applies to the `vector` argument
// only, while the `plaintexts` argument still gets n-1 rotations.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementCiphertextCiphertextBabyStepGiantStep(
    const T& giantSteppedOperand, const T& babySteppedOperand, int64_t period,
    int64_t steps, DerivedRotationIndexFn derivedRotationIndexFn) {
  using NodeTy = ArithmeticDagNode<T>;

  // Avoid replicating and re-extracting by simulating the extraction step by
  // just returning the single ciphertext.
  auto extractFunc = [](std::shared_ptr<NodeTy> babySteppedDag,
                        int64_t extractionIndex) { return babySteppedDag; };

  return implementBabyStepGiantStep<T>(giantSteppedOperand, babySteppedOperand,
                                       period, steps, extractFunc,
                                       derivedRotationIndexFn);
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
//
// This function produces a kernel using roughly n + 2sqrt(n) - 3 rotations
// (for matrix dimensions all order n), by applying the baby-step-giant-step
// method to reduce the number of rotations of packedA.
//
// This implements the BMM-I algorithm from https://eprint.iacr.org/2024/1762
// with modifications from LKAA25 (https://eprint.iacr.org/2025/1200):
//
//  - A simplification of the rotation formula in Sec 5.2.1 (equation 9).
//  - A baby-step-giant-step optimization of the summation below, from Sec
//    6.2.2 (equation 21).
//
// It computes
//
// C = sum_{c=0}^{n-1} rot(A, r1(c)) * rot(B, r2(c))
//
// where
//
//  r1(c) = cm
//  r2(c) = p(cm(p^{-1}) mod n) mod np
//
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBicyclicMatmul(const T& packedA, const T& packedB, int64_t m,
                        int64_t n, int64_t p) {
  APInt mAPInt = APInt(64, m);
  APInt nAPInt = APInt(64, n);
  APInt pAPInt = APInt(64, p);

  APInt mInvModN = multiplicativeInverse(mAPInt.urem(nAPInt), nAPInt);
  APInt pInvModN = multiplicativeInverse(pAPInt.urem(nAPInt), nAPInt);

  auto derivedRotationIndexFn = [&](int64_t giantStepSize,
                                    int64_t giantStepIndex,
                                    int64_t babyStepIndex, int64_t period) {
    APInt c(64, giantStepIndex * giantStepSize + babyStepIndex);
    APInt mAPInt(64, m);

    // RotY(c) = (p * (c * m * p^{-1} mod n)) mod (n * p)
    APInt rotyInner = (c * mAPInt * pInvModN.getSExtValue()).urem(nAPInt);
    APInt roty = (rotyInner * pAPInt).urem(nAPInt * pAPInt);

    APInt result = roty - APInt(64, period) * APInt(64, giantStepSize) *
                              APInt(64, giantStepIndex);
    return result.getSExtValue();
  };

  return implementCiphertextCiphertextBabyStepGiantStep<T>(
      packedA, packedB, /*period=*/m, /*steps=*/n, derivedRotationIndexFn);
}

// Returns an arithmetic DAG that implements the tricyclic batch matrix
// multiplication algorithm (ciphertext-ciphertext). Uses the tricyclic
// rotation formulas from LKAA25 (Tricycle paper) and applies BSGS to
// reduce rotations on the φ(A) side.
//
// The inputs packedA and packedB are expected to be tricyclic encodings
// φ(A) and φ(B) for tensors A ∈ R^{h×m×n} and B ∈ R^{h×n×p}. The function
// applies equation (22) and the ct-ct BSGS decomposition in Section 6.2.2.
//
// Parameters:
//  - packedA: tricyclic-encoded ciphertext for A (φ(A))
//  - packedB: tricyclic-encoded ciphertext for B (φ(B))
//  - h, m, n, p: tricyclic tensor dimensions (h: batch count / heads)
//
// Returns φ(Z) where Z = batch_matmul(A, B) as a DAG node.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementTricyclicBatchMatmul(const T& packedA, const T& packedB, int64_t h,
                              int64_t m, int64_t n, int64_t p) {
  APInt hAPInt = APInt(64, h);
  APInt mAPInt = APInt(64, m);
  APInt nAPInt = APInt(64, n);
  APInt pAPInt = APInt(64, p);

  APInt pInvModN = multiplicativeInverse(pAPInt.urem(nAPInt), nAPInt);

  APInt modulus = (hAPInt * nAPInt * pAPInt);
  // This follows Eq. (22) and the ct-ct BSGS decomposition.
  // RotY(c) = (h * p * ( (c * m * p^{-1}) mod n )) mod (h * n * p)
  auto derivedRotationIndexFn = [&](int64_t giantStepSize,
                                    int64_t giantStepIndex,
                                    int64_t babyStepIndex, int64_t period) {
    APInt c(64, giantStepIndex * giantStepSize + babyStepIndex);

    // rotyInner = (c * m * p^{-1}) mod n
    APInt rotyInner = (c * mAPInt * pInvModN.getSExtValue()).urem(nAPInt);

    // rotY calculation from LKAA25 Eq. (22):
    // RotY(c) = (h * p * (c * m * p^{-1} mod n)) mod (h * n * p)
    APInt roty = (rotyInner * hAPInt * pAPInt).urem(modulus);

    APInt result = roty - APInt(64, period) * APInt(64, giantStepSize) *
                              APInt(64, giantStepIndex);
    return result.getSExtValue();
  };

  int64_t period = h * m;
  return implementCiphertextCiphertextBabyStepGiantStep<T>(
      packedA, packedB, /*period=*/period, /*steps=*/n, derivedRotationIndexFn);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_KERNELIMPLEMENTATION_H_
