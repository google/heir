#ifndef LIB_DIALECT_RNS_IR_RNSOPS_H_
#define LIB_DIALECT_RNS_IR_RNSOPS_H_

#include <cstdint>

#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/APInt.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"    // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.h.inc"

namespace mlir {
namespace heir {
namespace rns {

template <typename Op>
LogicalResult verifyExtractSliceOp(Op* op, RNSType rnsType, int start,
                                   int size) {
  int64_t numLimbs = rnsType.getBasisTypes().size();

  if (start < 0) {
    return op->emitOpError()
           << "start index " << start << " cannot be negative";
  }

  if (size < 0) {
    return op->emitOpError() << "size " << size << " cannot be negative";
  }

  if (start + size > numLimbs) {
    return op->emitOpError()
           << "slice of size " << size << " starting at " << start
           << " is out of bounds for RNS type with " << numLimbs << " limbs";
  }

  return success();
}

template <typename Op>
RNSType inferExtractSliceReturnTypes(MLIRContext* ctx, Op* op,
                                     RNSType elementType) {
  int64_t start = op->getStart().getZExtValue();
  int64_t size = op->getSize().getZExtValue();
  return RNSType::get(
      ctx, elementType.getBasisTypes().drop_front(start).take_front(size));
}

// Given an RNS value x with limbs x_i = x mod q_i for pairwise-coprime moduli
// q_0, ..., q_{k-1}, compute its mixed-radix digits c_0, ..., c_{k-1} such
// that
//
//   x = c_0 + c_1 * q_0 + c_2 * (q_0 * q_1) + ... + c_{k-1} * (q_0 * ... *
//   q_{k-2}).
//
// Equivalently, if Q_i = \prod_{j=0}^i q_j, then each coefficient satisfies
//
//   c_i = (x_i - \sum_{j=0}^{i-1} c_j * Q_{j-1}) * Q_{i-1}^{-1} mod q_i,
//
// where qInvProds[i - 1] stores Q_{i-1}^{-1} in Z / (q_i Z). The returned SSA
// values are the c_i lifted from their limb-wise mod_arith types into the
// corresponding integer lowering types.
FailureOr<SmallVector<Value>> computeMixedRadixCoeffs(
    ImplicitLocOpBuilder& b, Value input, const ArrayAttr& qInvProds);

// For an input basis q_0, ..., q_{k-1}, build the precomputed inverses used by
// computeMixedRadixCoeffs and convertBasis. The returned array has length k - 1
// and stores
//
//   qInvProds[i] = (q_0 * ... * q_i)^{-1} mod q_{i+1}.
//
// Each entry is encoded as a mod_arith attribute in the corresponding
// q_{i+1}-limb type.
FailureOr<ArrayAttr> buildQInvProds(mlir::MLIRContext* ctx, RNSType basisTy);

// Given x \in Z_Q represented as {x_i mod q_i} where Q=\prod q_i,
// convertBasis returns the target-basis representation obtained by:
//  1. Compute x's centered integer representative in [-Q/2, Q/2]
//  2. Reduce the integer representation mod p_i for each p_i in the target
//  basis.
//
// This performs the conversion directly from the RNS representation of x,
// without first reconstructing x over the integers. Note that both bases must
// be odd.
FailureOr<Value> convertBasis(
    ImplicitLocOpBuilder b, ArrayAttr qInvProds, Value x, RNSType targetBasisTy,
    const llvm::DenseMap<APInt, size_t>& inputBasisIndexByModulus);

}  // namespace rns
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_RNS_IR_RNSOPS_H_
