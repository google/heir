#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"

#include <cassert>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "openfhe-muldepth-analysis"

namespace mlir {
namespace heir {

// Currently, this analysis targets the OpenFHE configuration.
// Since OpenFHE conservatively calculates multiplicative depth, it considers
// MulPlain and MulConst operations to consume a depth of 1. Therefore, the same
// approach has been applied here.

LogicalResult MulDepthAnalysis::visitOperation(
    Operation *op, ArrayRef<const MulDepthLattice *> operands,
    ArrayRef<MulDepthLattice *> results) {
  llvm::TypeSwitch<Operation &>(*op)
      .Case<openfhe::MulOp, openfhe::MulPlainOp, openfhe::MulConstOp,
            openfhe::MulNoRelinOp>([&](auto mulOp) {
        // In this case, 1 + (the maximum multiplicative depth among the
        // operands) becomes the multiplicative depth of the operation.
        LLVM_DEBUG(llvm::dbgs()
                   << "Visiting Mul: " << mulOp->getName() << "\n");
        // There should be only one result.
        assert(results.size() == 1);
        MulDepthLattice *r = results[0];
        MulDepth lhsMulDepth = operands[1]->getValue();
        if (lhsMulDepth.isInitialized()) {
          lhsMulDepth = MulDepth{lhsMulDepth.getValue() + 1};
        } else {
          // if lhs is not initialized, consider it as 0 (and then add 1).
          lhsMulDepth = MulDepth{1};
        }
        MulDepth rhsMulDepth = operands[2]->getValue();
        if (rhsMulDepth.isInitialized()) {
          rhsMulDepth = MulDepth{rhsMulDepth.getValue() + 1};
        } else {
          // if rhs is not initialized, consider it as 0 (and then add 1).
          rhsMulDepth = MulDepth{1};
        }
        LLVM_DEBUG({
          llvm::dbgs() << "lhsMulDepth: " << lhsMulDepth << "\n";
          llvm::dbgs() << "rhsMulDepth: " << rhsMulDepth << "\n";
        });
        ChangeResult result = r->join(MulDepth::join(lhsMulDepth, rhsMulDepth));
        propagateIfChanged(r, result);
        LLVM_DEBUG({
          llvm::dbgs() << "MulDepth: " << results[0]->getValue() << "\n";
        });
      })
      .Default([&](Operation &defaultOp) {
        // In this case, the maximum multiplicative depth among the operands
        // becomes the multiplicative depth of the operation.
        LLVM_DEBUG(
            { llvm::dbgs() << "Visiting: " << defaultOp.getName() << "\n"; });
        for (const MulDepthLattice *operand : operands) {
          for (MulDepthLattice *r : results) {
            ChangeResult result = r->join(*operand);
            propagateIfChanged(r, result);
          }
        }
      });
  return mlir::success();
}

}  // namespace heir
}  // namespace mlir
