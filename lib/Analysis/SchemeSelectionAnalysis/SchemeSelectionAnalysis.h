#ifndef LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_
#define LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "lib/Analysis/SchemeInfoAnalysis/SchemeInfoAnalysis.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

constexpr StringRef BGV = "bgv";
constexpr StringRef BFV = "bfv";
constexpr StringRef CKKS = "ckks";
constexpr StringRef CGGI = "cggi";

class SchemeSelectionAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<SchemeInfoLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const SchemeInfoLattice *> operands,
                               ArrayRef<SchemeInfoLattice *> results) override;

  void setToEntryState(SchemeInfoLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(NatureOfComputation()));
  }
};

std::string annotateModuleWithScheme(Operation *top, DataFlowSolver *solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_
