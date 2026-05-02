#include "lib/Dialect/Rotom/Transforms/SeedLayout/SeedLayout.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/StringRef.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
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

struct SeedLayout : public impl::SeedLayoutBase<SeedLayout> {
  using SeedLayoutBase::SeedLayoutBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = module.getContext();

    DataFlowSolver solver;
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

    for (Value v : valuesToSeed) {
      // FIXME: cache the seeds
      auto tensorType = dyn_cast<RankedTensorType>(v.getType());
      if (auto secretType = dyn_cast<secret::SecretType>(v.getType())) {
        tensorType = dyn_cast<RankedTensorType>(secretType.getValueType());
      }
      if (!tensorType) continue;

      LLVM_DEBUG(llvm::dbgs() << "Seeding value: " << v << "\n");

      ArrayRef<int64_t> shape = tensorType.getShape();
      int64_t rank = shape.size();

      SmallVector<int64_t> dims(shape.begin(), shape.end());

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
          // slots. If tuple[d] < shape[d], we need a vector dimension of size
          // shape[d]/tuple[d] and stride tuple[d] (to skip over elements in
          // slots). These provide the ciphertext segmentation.
          SmallVector<Attribute> ctDims;
          for (int64_t d = 0; d < rank; ++d) {
            if (tuple[d] < shape[d]) {
              ctDims.push_back(
                  DimAttr::get(ctx, d, shape[d] / tuple[d], tuple[d]));
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

      LLVM_DEBUG(llvm::dbgs()
                 << "\t- Found " << layouts.size() << " layouts for value\n");

      if (!layouts.empty()) {
        auto seedAttr = SeedAttr::get(ctx, ArrayAttr::get(ctx, layouts));
        setAttributeAssociatedWith(v, "rotom.seed", seedAttr);
      }
    }
  }
};

}  // namespace mlir::heir::rotom
