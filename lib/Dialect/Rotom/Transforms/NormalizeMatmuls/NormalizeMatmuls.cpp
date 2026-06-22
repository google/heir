#include "lib/Dialect/Rotom/Transforms/NormalizeMatmuls/NormalizeMatmuls.h"

#include <cstdint>

#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

#define GEN_PASS_DEF_NORMALIZEMATMULS
#include "lib/Dialect/Rotom/Transforms/NormalizeMatmuls/NormalizeMatmuls.h.inc"

namespace {

// Rewrites a vector-times-matrix matmul x(1xK) * M(KxN) (N > 1) into
// (M^T * x^T)^T, i.e. transpose(matmul(M^T, x^T)). The ciphertext-axis diagonal
// matvec kernel only handles matrix * column-vector (M'xK * Kx1), and a neural
// net layer y = x . W^T is exactly this transposed orientation: rewriting it to
// W . x^T lets the existing kernel apply, with the result transposed back. When
// M is itself a transpose (the common y = x . W^T case) its input is used
// directly, so no redundant transpose-of-transpose is created. Must run before
// seeding so the seeds land on the rewritten operands.
void normalizeRowVectorMatmuls(ModuleOp module) {
  SmallVector<linalg::MatmulOp> targets;
  module.walk([&](linalg::MatmulOp op) {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1) return;
    auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2) {
      return;
    }
    if (lhsType.getDimSize(0) == 1 && rhsType.getDimSize(1) > 1) {
      targets.push_back(op);
    }
  });

  for (linalg::MatmulOp op : targets) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value x = op.getInputs()[0];      // 1 x K
    Value m = op.getInputs()[1];      // K x N
    Value init = op.getOutputs()[0];  // 1 x N
    Type elt = cast<RankedTensorType>(x.getType()).getElementType();
    int64_t kDim = cast<RankedTensorType>(x.getType()).getDimSize(1);
    int64_t nDim = cast<RankedTensorType>(m.getType()).getDimSize(1);

    auto transpose = [&](Value v, int64_t rows, int64_t cols) -> Value {
      auto empty = tensor::EmptyOp::create(builder, loc,
                                           ArrayRef<int64_t>{rows, cols}, elt);
      auto t = linalg::TransposeOp::create(builder, loc, v, empty,
                                           ArrayRef<int64_t>{1, 0});
      return t.getOperation()->getResult(0);
    };

    // M^T (N x K): reuse the pre-transpose input when M = transpose(W), but only
    // for a genuine [1, 0] transpose. An identity ([0, 1]) permutation leaves the
    // operand in K x N orientation; reusing it would build matmul(K x N, K x 1),
    // an invalid contraction. Fall back to an explicit transpose in that case.
    Value matrix;
    auto producer = m.getDefiningOp<linalg::TransposeOp>();
    if (producer && producer.getPermutation().size() == 2 &&
        producer.getPermutation()[0] == 1 && producer.getPermutation()[1] == 0) {
      matrix = producer.getInput();
    } else {
      matrix = transpose(m, nDim, kDim);
    }
    Value xCol = transpose(x, kDim, 1);        // 1 x K -> K x 1
    Value initCol = transpose(init, nDim, 1);  // 1 x N -> N x 1

    Type colType = RankedTensorType::get({nDim, 1}, elt);
    auto matmul =
        linalg::MatmulOp::create(builder, loc, TypeRange{colType},
                                 ValueRange{matrix, xCol}, ValueRange{initCol});
    Value yCol = matmul.getOperation()->getResult(0);  // N x 1
    Value y = transpose(yCol, 1, nDim);                // N x 1 -> 1 x N
    op.getOperation()->getResult(0).replaceAllUsesWith(y);
    op.erase();
  }
}

struct NormalizeMatmuls
    : public impl::NormalizeMatmulsBase<NormalizeMatmuls> {
  using NormalizeMatmulsBase::NormalizeMatmulsBase;

  void runOnOperation() override {
    normalizeRowVectorMatmuls(getOperation());
  }
};

}  // namespace
}  // namespace rotom
}  // namespace heir
}  // namespace mlir
