#ifndef LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_
#define LIB_ANALYSIS_SCHEMESELECTIONANALYSIS_SCHEMESELECTIONANALYSIS_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "llvm/include/llvm/Support/raw_ostream.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                      // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                          // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                      // from @llvm-project

namespace mlir {
namespace heir {

constexpr StringRef numBoolOpsAttrName = "natcomp.boolOps";
constexpr StringRef numBitOpsAttrName = "natcomp.bitOps";
constexpr StringRef numIntArithOpsAttrName = "natcomp.intOps";
constexpr StringRef numRealArithOpsAttrName = "natcomp.realOps";
constexpr StringRef numCmpOpsAttrName = "natcomp.cmpOps";
constexpr StringRef numNonLinOpsAttrName = "natcomp.nonLinOps";
constexpr StringRef boolOpsTypeName = "bool";
constexpr StringRef bitOpsTypeName = "bit";
constexpr StringRef intArithOpsTypeName = "intArith";
constexpr StringRef realArithOpsTypeName = "realArith";
constexpr StringRef cmpOpsTypeName = "cmp";
constexpr StringRef nonLinOpsTypeName = "nonLin";

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
        numCmpOps(numCmpOps),
        numNonLinOps(numNonLinOps) {}

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

  int getNonLinOpsCount() const {
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

 NatureOfComputation operator+(const NatureOfComputation &rhs) const {
	if (!isInitialized() && !rhs.isInitialized()) {
      return *this; // return the current object
    }

    if (isInitialized() && !rhs.isInitialized()) {
      return *this; // return the current object
    }

    if (!isInitialized() && rhs.isInitialized()) {
      return rhs; // return the rhs object
    }

    // Both are initialized
    return NatureOfComputation(
        numBoolOps + rhs.numBoolOps,
        numBitOps + rhs.numBitOps,
        numIntArithOps + rhs.numIntArithOps,
        numRealArithOps + rhs.numRealArithOps,
        numCmpOps + rhs.numCmpOps,
        numNonLinOps + rhs.numNonLinOps);
  }

  StringRef getDominantAttributeName() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    //TODO: define what happens when all are equal
    int maxCount = numBoolOps;
    StringRef attributeName = numBoolOpsAttrName;

    if (numBitOps > maxCount) {
        maxCount = numBitOps;
        attributeName = numBitOpsAttrName;
    }
    if (numIntArithOps > maxCount) {
        maxCount = numIntArithOps;
        attributeName = numIntArithOpsAttrName;
    }
    if (numRealArithOps > maxCount) {
        maxCount = numRealArithOps;
        attributeName = numRealArithOpsAttrName;
    }
    if (numCmpOps > maxCount) {
        maxCount = numCmpOps;
        attributeName = numCmpOpsAttrName;
    }
    if (numNonLinOps > maxCount) {
        maxCount = numNonLinOps;
        attributeName = numNonLinOpsAttrName;
    }

    return attributeName;
}

int getDominantComputationCount() const {
    assert(isInitialized() && "NatureOfComputation not initialized");
    
    int maxCount = numBoolOps;

    if (numBitOps > maxCount) {
        maxCount = numBitOps;
    }
    if (numIntArithOps > maxCount) {
        maxCount = numIntArithOps;
    }
    if (numRealArithOps > maxCount) {
        maxCount = numRealArithOps;
    }
    if (numCmpOps > maxCount) {
        maxCount = numCmpOps;
    }
    if (numNonLinOps > maxCount) {
        maxCount = numNonLinOps;
    }

    return maxCount; // Returns the highest count after all comparisons
}

  static NatureOfComputation max(const NatureOfComputation &lhs,
                                 const NatureOfComputation &rhs) {
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
         << "; numBitOps=" << numBitOps << "; numIntArithOps=" << numIntArithOps
         << "; numRealArithOps=" << numRealArithOps << "; numCmpOps=" << numCmpOps
         << "; numNonLinOps=" << numNonLinOps << ")";
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
    : public dataflow::SparseForwardDataFlowAnalysis<SchemeInfoLattice>{
 private:
  NatureOfComputation counter;
 public:
  SchemeSelectionAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis<SchemeInfoLattice>(solver), counter(0, 0, 0, 0, 0, 0) {}
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  
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
