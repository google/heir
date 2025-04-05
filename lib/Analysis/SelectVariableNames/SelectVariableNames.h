#ifndef LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
#define LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_

#include <cassert>
#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

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
    return variableNames.lookup(value);
  }

  // Return the unique integer assigned to a given value.
  int getIntForValue(Value value) const {
    assert(variableToInteger.contains(value));
    return variableToInteger.lookup(value);
  }

  // Map from one value's name to another value's name.
  void mapValueNameToValue(Value fromValue, Value toValue) {
    assert(variableNames.contains(toValue));
    variableNames[fromValue] = variableNames[toValue];
    variableToInteger[fromValue] = variableToInteger[toValue];
  }

  bool contains(Value value) const { return variableNames.contains(value); }

 private:
  std::string suggestNameForValue(Value value);

  std::string defaultPrefix{"v"};
  llvm::DenseMap<Value, std::string> variableNames;
  llvm::DenseMap<Value, int> variableToInteger;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
