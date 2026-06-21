#include "lib/Dialect/Rotom/Transforms/SeedLayout/SeedLayout.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

namespace mlir::heir::rotom {

namespace {

/// findValidTuples recursively finds all valid tuples (i_0, ..., i_k) such that
/// the product of all i_j equals n, where 1 <= i_j <= d_j (dim size) and i_j is
/// a power of 2. These tuples represent the size of each tensor dimension
/// packed into the ciphertext slots.
void findValidTuples(ArrayRef<int64_t> dims, int64_t n, int64_t currentProd,
                     SmallVectorImpl<int64_t>& currentTuple,
                     SmallVectorImpl<SmallVector<int64_t>>& validTuples) {
  if (currentTuple.size() == dims.size()) {
    if (currentProd == n) {
      validTuples.push_back(
          SmallVector<int64_t>(currentTuple.begin(), currentTuple.end()));
    }
    return;
  }

  int64_t dimIdx = currentTuple.size();
  int64_t maxVal = dims[dimIdx];

  for (int64_t i = 1; i <= maxVal; i *= 2) {
    if (n % (currentProd * i) != 0) continue;
    currentTuple.push_back(i);
    findValidTuples(dims, n, currentProd * i, currentTuple, validTuples);
    currentTuple.pop_back();
  }
}
}  // namespace

#define DEBUG_TYPE "rotom-seed-layout"

#define GEN_PASS_DEF_SEEDLAYOUT
#include "lib/Dialect/Rotom/Transforms/SeedLayout/SeedLayout.h.inc"

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

    // M^T (N x K): reuse the pre-transpose input when M = transpose(W).
    Value matrix;
    if (auto producer = m.getDefiningOp<linalg::TransposeOp>()) {
      matrix = producer.getInput();
    } else {
      matrix = transpose(m, nDim, kDim);
    }
    Value xCol = transpose(x, kDim, 1);      // 1 x K -> K x 1
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

struct SeedLayout : public impl::SeedLayoutBase<SeedLayout> {
  using SeedLayoutBase::SeedLayoutBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = module.getContext();

    // Normalize x . W^T layers to the matrix * column-vector orientation the
    // diagonal matvec kernel handles, before seeding the (rewritten) operands.
    normalizeRowVectorMatmuls(module);

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      signalPassFailure();
      return;
    }

    DenseSet<Value> valuesToSeed;

    // Seed secret function arguments
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.empty()) return;
      Block& entryBlock = funcOp.front();
      for (Value arg : entryBlock.getArguments()) {
        if (auto secretType = dyn_cast<secret::SecretType>(arg.getType())) {
          if (isa<RankedTensorType>(secretType.getValueType())) {
            valuesToSeed.insert(arg);
          }
        }
      }
    });

    // Seed cleartext values immediately used in secret computation
    module.walk([&](Operation* op) {
      auto genericOp = op->getParentOfType<secret::GenericOp>();
      if (genericOp) {
        for (Value operand : op->getOperands()) {
          if (!isSecret(operand, &solver) &&
              dyn_cast<RankedTensorType>(operand.getType())) {
            valuesToSeed.insert(operand);
          }
        }
      }
    });

    // Create cache for seed layouts to avoid recomputing seeds for equivalent
    // tensor types.
    DenseMap<Type, SeedAttr> seedCache;

    for (Value v : valuesToSeed) {
      auto tensorType = dyn_cast<RankedTensorType>(v.getType());
      if (auto secretType = dyn_cast<secret::SecretType>(v.getType())) {
        tensorType = dyn_cast<RankedTensorType>(secretType.getValueType());
      }
      if (!tensorType) continue;

      if (seedCache.contains(tensorType)) {
        setAttributeAssociatedWith(v, "rotom.seed", seedCache[tensorType]);
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Seeding value: " << v << "\n");

      ArrayRef<int64_t> shape = tensorType.getShape();
      int64_t rank = shape.size();

      SmallVector<int64_t> dims;
      dims.reserve(rank);
      for (int64_t d : shape) {
        dims.push_back(nextPowerOfTwo(d));
      }

      // Find all valid tuples (i_0, ..., i_k) such that the product of all i_j
      // equals n, where 1 <= i_j <= d_j (dim size) and i_j is a power of 2.
      // These tuples represent the size of each tensor dimension packed into
      // the ciphertext slots.
      SmallVector<SmallVector<int64_t>> validTuples;
      SmallVector<int64_t> currentTuple;  // scratch space for recursion
      findValidTuples(dims, n, 1, currentTuple, validTuples);

      SmallVector<Attribute> layouts;
      for (const auto& tuple : validTuples) {
        // Generate all permutations (orders) for the dimensions. This will
        // explore different orderings (row-major and column-major) of axis
        // traversal.
        SmallVector<int64_t> perm(rank);
        std::iota(perm.begin(), perm.end(), 0);

        do {
          // Generate slot dimensions based on the current permutation.
          // tuple[d] gives the size of dimension d in slots.
          SmallVector<Attribute> slotDims;
          for (int64_t d : perm) {
            slotDims.push_back(DimAttr::get(ctx, d, tuple[d], 1));
          }

          // Generate vector dimensions for dimensions that don't fully fit in
          // slots. If tuple[d] < dims[d], we need a vector dimension of size
          // dims[d]/tuple[d] and stride tuple[d] (to skip over elements in
          // slots). These provide the ciphertext segmentation.
          SmallVector<Attribute> ctDims;
          for (int64_t d = 0; d < rank; ++d) {
            if (tuple[d] < dims[d]) {
              ctDims.push_back(
                  DimAttr::get(ctx, d, dims[d] / tuple[d], tuple[d]));
            }
          }

          // Construct the final list of dimensions. Place ctDims (vector dims)
          // first for segmentation and slotDims last.
          SmallVector<Attribute> allDims = ctDims;
          allDims.append(slotDims);

          layouts.push_back(
              LayoutAttr::get(ctx, ArrayAttr::get(ctx, allDims), n));
        } while (std::next_permutation(perm.begin(), perm.end()));
      }

      // If the padded tensor is smaller than the ciphertext, no tuple reaches
      // product == n (findValidTuples found nothing): pack every dimension fully
      // into slots and replicate the packed block to fill the remaining
      // capacity. Without this a small operand -- e.g. an MNIST 10x512 weight or
      // a 512-length activation at ciphertext size 32768 -- gets no seed, so the
      // layout search has no candidate and the matmul kernel is never assigned.
      if (layouts.empty()) {
        int64_t fullProd = 1;
        for (int64_t d : dims) fullProd *= d;
        if (fullProd > 0 && fullProd < n && n % fullProd == 0) {
          int64_t replicationFactor = n / fullProd;
          SmallVector<int64_t> perm(rank);
          std::iota(perm.begin(), perm.end(), 0);
          do {
            // The replication dim is outermost (block replication): the packed
            // block occupies slots [0, fullProd) and repeats every fullProd.
            SmallVector<Attribute> allDims;
            allDims.push_back(
                DimAttr::get(ctx, /*dim=*/-1, replicationFactor, fullProd));
            for (int64_t d : perm) {
              allDims.push_back(DimAttr::get(ctx, d, dims[d], 1));
            }
            layouts.push_back(
                LayoutAttr::get(ctx, ArrayAttr::get(ctx, allDims), n));
          } while (std::next_permutation(perm.begin(), perm.end()));
        }
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "\t- Found " << layouts.size() << " layouts for value\n");

      if (!layouts.empty()) {
        auto seedAttr = SeedAttr::get(ctx, ArrayAttr::get(ctx, layouts));
        setAttributeAssociatedWith(v, "rotom.seed", seedAttr);
        seedCache[tensorType] = seedAttr;
      }
    }
  }
};

}  // namespace mlir::heir::rotom
