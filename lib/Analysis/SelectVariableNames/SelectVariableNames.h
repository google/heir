#ifndef LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
#define LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_

#include <cassert>
#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

class VariableName {
 public:
  VariableName() = default;
  VariableName(const std::string &name, int integer)
      : initialized(true), name(name), integer(integer) {}
  ~VariableName() = default;

  bool operator==(const VariableName &other) const {
    return initialized == other.initialized && name == other.name &&
           integer == other.integer;
  }

  // For Lattice in DataFlowFramework
  static VariableName join(const VariableName &lhs, const VariableName &rhs) {
    if (!lhs.isInitialized() || lhs.getName().empty()) return rhs;
    if (!rhs.isInitialized() || rhs.getName().empty()) return lhs;

    if (lhs.getName() == rhs.getName()) return lhs;

    // If the names are different, we can just return the first one
    return lhs;
  }

  void print(raw_ostream &os) const {
    os << "Name: " << name << " Int: " << integer;
  }

  bool isInitialized() const { return initialized; }
  std::string getName() const { return name; }
  int getInteger() const { return integer; }

 private:
  bool initialized = false;
  std::string name;
  int integer;
};

class SelectVariableNames {
 public:
  SelectVariableNames(Operation *op);
  ~SelectVariableNames() = default;

  /// Return the name assigned to the given value, or an empty string if the
  /// value was not assigned a name (suggesting the value was not in the IR
  /// tree that this class was constructed with).
  std::string getNameForValue(Value value) const {
    return lookup(value).getName();
  }

  /// Return the unique integer assigned to a given value.
  int getIntForValue(Value value) const { return lookup(value).getInteger(); }

  /// returns the internal VariableName struct
  VariableName lookup(Value value) const {
    assert(variableNames.contains(value));
    return variableNames.lookup(value);
  }

  // Map from one value's name to another value's name.
  void mapValueNameToValue(Value fromValue, Value toValue) {
    assert(variableNames.contains(toValue));
    variableNames[fromValue] = variableNames[toValue];
  }

  bool contains(Value value) const { return variableNames.contains(value); }

 private:
  std::string suggestPrefixForValue(Value value);

  std::string defaultPrefix{"v"};
  llvm::DenseMap<Value, VariableName> variableNames;
};

//===----------------------------------------------------------------------===//
// DataFlowAnalysis Wrapper
//===----------------------------------------------------------------------===//

class VariableNameLattice : public dataflow::Lattice<VariableName> {
 public:
  using Lattice::Lattice;
};

/// This analysis does not update the Lattice when IR changes!
class SelectVariableNameAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<VariableNameLattice> {
 public:
  explicit SelectVariableNameAnalysis(
      DataFlowSolver &solver, const SelectVariableNames &selectVariableNames)
      : SparseForwardDataFlowAnalysis(solver),
        selectVariableNames(selectVariableNames) {}

  void setToEntryState(VariableNameLattice *lattice) override {
    propagateIfChanged(
        lattice,
        lattice->join(selectVariableNames.lookup(lattice->getAnchor())));
  }

  // does nothing
  LogicalResult visitOperation(
      Operation *operation, ArrayRef<const VariableNameLattice *> operands,
      ArrayRef<VariableNameLattice *> results) override {
    return success();
  };

 private:
  SelectVariableNames selectVariableNames;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
