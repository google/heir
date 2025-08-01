#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "SchemeSelectionAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "SchemeSelection"

namespace mlir {
namespace heir {

LogicalResult SchemeSelectionAnalysis::visitOperation(
    Operation *op, ArrayRef<const SchemeInfoLattice *> operands,
    ArrayRef<SchemeInfoLattice *> results) {
  LLVM_DEBUG(llvm::dbgs()
      << "Visiting: " << op->getName() << ". ");

  
  return success();
}

void annotateModuleWithScheme(Operation *top, DataFlowSolver *solver) {

    LLVM_DEBUG(llvm::dbgs()
        << "Top operation is: " << top->getName() << "\n");

  }

}  // namespace heir
}  // namespace mlir