#ifndef LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_
#define LIB_ANALYSIS_SELECTVARIABLENAMES_SELECTVARIABLENAMES_H_

#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project

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
    if (auto constantOp =
            dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
      auto valueAttr = constantOp.getValue();
      if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
        return std::to_string(intAttr.getInt());
      } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
        SmallString<128> strValue;
        auto apValue = APFloat(floatAttr.getValueAsDouble());
        apValue.toString(strValue, /*FormatPrecision=*/0,
                         /*FormatMaxPadding=*/10,
                         /*TruncateZero=*/true);
        return std::string(strValue);
      }
    }
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
