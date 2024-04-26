#ifndef LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
#define LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_

#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project

namespace mlir {
namespace heir {

class SelectVariableNames {
 public:
  SelectVariableNames(Operation *op);
  ~SelectVariableNames() = default;

  /// Return the name assigned to the given value, or an empty string if the
  /// value was not assigned a name (suggesting the value was not in the IR
  /// tree that this class was constructed with).
  std::string getNameForValue(Value value) const {
    assert(variableNames.contains(value));
    return prefix + std::to_string(variableNames.lookup(value));
  }

  // Return the unique integer assigned to a given value.
  int getIntForValue(Value value) const {
    assert(variableNames.contains(value));
    return variableNames.lookup(value);
  }

 private:
  llvm::DenseMap<Value, int> variableNames;

  std::string prefix{"v"};
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
