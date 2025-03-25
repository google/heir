#ifndef LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_
#define LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
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

class NatureOfComputation {
 public:
  NatureOfComputation()
      : initialized(false),
        numBoolOps(0),
        numBitOps(0),
        numIntArithOps(0),
        numRealArithOps(0),
        numCmpOps(0),
        numNonLinOps(0) {}

  explicit NatureOfComputation(int numBoolOps, int numBitOps,
                               int numIntArithOps, int numRealArithOps,
                               int numCmpOps, int numNonLinOps)
      : initialized(true),
        numBoolOps(numBoolOps),
        numBitOps(numBitOps),
        numIntArithOps(numIntArithOps),
        numRealArithOps(numRealArithOps),
        numCmpOps(numBoolOps),
        numNonLinOps(numBitOps) {}

  void countOperation(Operation *op);

  int getBoolOpsCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    return numBoolOps;
  }

  int getBitOpsCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    return numBitOps;
  }

  int getIntArithOpsCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    return numIntArithOps;
  }

  int getRealArithOpsCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    return numRealArithOps;
  }

  int getCmpOpsCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    return numCmpOps;
  }

  int getnumNonLinOpsCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    return numNonLinOps;
  }

  bool isInitialized() const { return initialized; }

  bool operator==(const NatureOfComputation &rhs) const {
    return initialized == rhs.initialized && numBoolOps == rhs.numBoolOps &&
           numBitOps == rhs.numBitOps && numIntArithOps == rhs.numIntArithOps &&
           numRealArithOps == rhs.numRealArithOps &&
           numCmpOps == rhs.numCmpOps && numNonLinOps == rhs.numNonLinOps;
  }

  static NatureOfComputation max(const NatureOfComputation &lhs,
                                 const NatureOfComputation &rhs) {
    assert(lhs.isInitialized() && rhs.isInitialized() &&
           "NatureOfComputation not initialized");
    return NatureOfComputation(
        std::max(lhs.numBoolOps, rhs.numBoolOps),
        std::max(lhs.numBitOps, rhs.numBitOps),
        std::max(lhs.numIntArithOps, rhs.numIntArithOps),
        std::max(lhs.numRealArithOps, rhs.numRealArithOps),
        std::max(lhs.numCmpOps, rhs.numCmpOps),
        std::max(lhs.numNonLinOps, rhs.numNonLinOps));
  }

  static NatureOfComputation join(const NatureOfComputation &lhs,
                                  const NatureOfComputation &rhs) {
    if (!lhs.isInitialized()) {
      return rhs;
    }

    if (!rhs.isInitialized()) {
      return lhs;
    }

    return NatureOfComputation::max(lhs, rhs);
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "NatureOfComputation(numBoolOps=" << numBoolOps
         << "numBitOps=" << numBitOps << "numIntArithOps=" << numIntArithOps
         << "numRealArithOps=" << numRealArithOps << "numCmpOps=" << numCmpOps
         << "numNonLinOps=" << numNonLinOps << ")";
    } else {
      os << "NatureOfComputation(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const NatureOfComputation &state) {
    state.print(os);
    return os;
  }

 private:
  bool initialized;
  int numBoolOps;
  int numBitOps;
  int numIntArithOps;
  int numRealArithOps;
  int numCmpOps;
  int numNonLinOps;
};

class SchemeInfoLattice : public dataflow::Lattice<NatureOfComputation> {
 public:
  using Lattice::Lattice;
};

class SchemeSelectionAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<SchemeInfoLattice>,
      public SecretnessAnalysisDependent<SchemeSelectionAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<SchemeSelectionAnalysis>;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const SchemeInfoLattice *> operands,
                               ArrayRef<SchemeInfoLattice *> results) override;

  void setToEntryState(SchemeInfoLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(NatureOfComputation()));
  }
};

void annotateNatureOfComputation(Operation *top, DataFlowSolver *solver,
                                 int baseLevel = 0);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_
