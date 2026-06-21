#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_

#include <cstdint>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

class RotomTensorOpLowering {
 public:
  explicit RotomTensorOpLowering(const ContextAwareTypeConverter* typeConverter)
      : typeConverter(typeConverter) {}

  LogicalResult lowerMatmul(
      linalg::MatmulOp op, linalg::MatmulOp::Adaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const;

  LogicalResult lowerElementwiseBinary(
      Operation* op, Value originalResult, ValueRange adaptorOperands,
      ContextAwareConversionPatternRewriter& rewriter) const;

 private:
  tensor_ext::LayoutAttr getLayoutAttr(Value value) const;

  FailureOr<std::vector<std::vector<int64_t>>> getRangePointsForDomain(
      tensor_ext::LayoutAttr layout, ArrayRef<int64_t> domain) const;

  // Same as above but reuses an already-built IntegerRelation. Building the
  // relation (LayoutAttr::getIntegerRelation serializes the layout to an ISL
  // string and re-parses it) is the dominant cost, so a caller that queries
  // many domain points of one layout -- e.g. the matvec kernels verifying every
  // matrix element -- builds the relation once and reuses it here, turning an
  // O(m*n) ISL rebuild into O(1).
  FailureOr<std::vector<std::vector<int64_t>>> getRangePointsForDomain(
      const presburger::IntegerRelation& relation,
      ArrayRef<int64_t> domain) const;

  FailureOr<Value> createMaskForPoints(
      RankedTensorType ciphertextSemanticType,
      const std::vector<std::vector<int64_t>>& points,
      ImplicitLocOpBuilder& b) const;

  FailureOr<DenseIntElementsAttr> createRemapAttr(
      MLIRContext* ctx, const std::vector<std::vector<int64_t>>& sourcePoints,
      const std::vector<std::vector<int64_t>>& targetPoints) const;

  FailureOr<Value> alignDomainPointToOutput(
      Value source, RankedTensorType ciphertextSemanticType,
      tensor_ext::LayoutAttr sourceLayout, ArrayRef<int64_t> sourceDomain,
      ArrayRef<int64_t> outputDomain, tensor_ext::LayoutAttr outputLayout,
      ImplicitLocOpBuilder& b) const;

  // Attempts the rotate-multiply-accumulate matmul kernel: groups every scalar
  // contribution by the ciphertext rotation that realizes it and emits one
  // masked rotate-product per group. Returns success (having replaced `op`)
  // when applicable, or failure (emitting nothing) so the caller can fall back
  // to the brute-force per-scalar lowering. Currently limited to single-
  // ciphertext layouts with injective slot packings.
  LogicalResult lowerMatmulByRotation(
      linalg::MatmulOp op, Value lhs, Value rhs, Value output,
      RankedTensorType ciphertextSemanticType, tensor_ext::LayoutAttr lhsLayout,
      tensor_ext::LayoutAttr rhsLayout, tensor_ext::LayoutAttr outputLayout,
      int64_t m, int64_t n, int64_t p,
      ContextAwareConversionPatternRewriter& rewriter) const;

  Value createRotate(Value tensor, int64_t shift, ImplicitLocOpBuilder& b) const;

  // Lowers a matvec A(m x n) * x(n x 1) -> out(m x 1) when the matrix is packed
  // as a ciphertext-axis diagonal (one ciphertext per diagonal, A[i,k] at
  // ct = (i - k) mod M, slot = k) and the vector/output occupy a single
  // ciphertext (slot = index). Emits the Halevi-Shoup diagonal kernel with a
  // baby-step/giant-step schedule over the M = numCiphertexts diagonals, then a
  // residual rotate-and-sum over the K/M column blocks:
  //   S   = sum_g rotate( sum_b rotate(x,-b) * rotate(diag_{b+B*g},-b), -B*g )
  //   out = init + mask( rotateAndSum(S, K/M, M) )
  // M == K is the square case (no residual); M < K (M | K) is the squat case.
  // ~2*sqrt(M) ciphertext-vector rotations + log2(K/M) residual; the per-diagonal
  // matrix rotations fold away for a plaintext matrix. Returns success (op
  // replaced) when applicable, else failure to fall back. Currently single-
  // period: numSlots == K == n (larger ciphertexts need period-K replication).
  LogicalResult lowerMatvecCtDiagonalBsgs(
      linalg::MatmulOp op, TypedValue<RankedTensorType> lhs,
      TypedValue<RankedTensorType> rhs, TypedValue<RankedTensorType> output,
      tensor_ext::LayoutAttr lhsLayout, tensor_ext::LayoutAttr rhsLayout,
      tensor_ext::LayoutAttr outputLayout, int64_t m, int64_t n, int64_t p,
      ContextAwareConversionPatternRewriter& rewriter) const;

  // Densely-packed diagonal matvec: the N-slot matrix ciphertext holds P = N/K
  // diagonals (each a K-slot block) instead of one, so A is M/P ciphertexts
  // (diagonal d at ct = floor(d/P), slot block (d mod P)*K). The same baby-step/
  // giant-step schedule runs K-wide (the vector and a K-block diagonal are
  // extracted), and the K-wide result is inserted into the N-slot output. This
  // is the dense (slot-efficient) packing the straddle materializer enables;
  // P == 1 is the single-period kernel above. Returns success (op replaced) when
  // applicable, else failure to fall back.
  LogicalResult lowerMatvecDenseDiagonal(
      linalg::MatmulOp op, TypedValue<RankedTensorType> lhs,
      TypedValue<RankedTensorType> rhs, TypedValue<RankedTensorType> output,
      tensor_ext::LayoutAttr lhsLayout, tensor_ext::LayoutAttr rhsLayout,
      tensor_ext::LayoutAttr outputLayout, int64_t m, int64_t n, int64_t p,
      ContextAwareConversionPatternRewriter& rewriter) const;

  const ContextAwareTypeConverter* typeConverter;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_ROTOMTENSOROPLOWERING_H_
