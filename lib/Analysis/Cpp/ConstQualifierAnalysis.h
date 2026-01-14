#ifndef LIB_ANALYSIS_CPP_CONSTQUALIFIERANALYSIS_H_
#define LIB_ANALYSIS_CPP_CONSTQUALIFIERANALYSIS_H_

#include <cassert>

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

class ConstQualifierAnalysis {
 public:
  ConstQualifierAnalysis(Operation* op);
  ~ConstQualifierAnalysis() = default;

  /// Return true if the value can be marked as const in C++ codegen.
  bool canBeMarkedConst(Value value) const {
    return constValues.contains(value);
  }

 private:
  llvm::DenseSet<Value> constValues;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_CPP_CONSTQUALIFIERANALYSIS_H_
