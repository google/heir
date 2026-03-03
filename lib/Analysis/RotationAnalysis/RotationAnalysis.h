#ifndef LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_
#define LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_

#include <cstdint>

#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project

namespace mlir {
namespace heir {

/// Analyzes an IR to determine a static set of indices used by rotation ops.
/// Requires sparse conditional constant propagation (-sccp) to be run on the
/// IR to guarantee constants are propagated and can be detected properly.
class RotationAnalysis {
 public:
  RotationAnalysis() {}

  const DenseSet<int64_t>& getRotationIndices() const {
    return rotationIndices;
  }

  LogicalResult run(Operation* op);

 private:
  LogicalResult handleScfFor(scf::ForOp forOp);
  LogicalResult analyzeRotationOp(RotationOpInterface rotationOp);

  bool wasVisited(RotationOpInterface op) {
    return visitedRotationOps.contains(op.getOperation());
  }
  bool wasVisited(Operation* op) { return visitedRotationOps.contains(op); }

  void markVisited(RotationOpInterface op) {
    visitedRotationOps.insert(op.getOperation());
  }

  void markVisited(Operation* op) { visitedRotationOps.insert(op); }

  DenseSet<int64_t> rotationIndices;
  DenseSet<Operation*> visitedRotationOps;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONANALYSIS_H_
