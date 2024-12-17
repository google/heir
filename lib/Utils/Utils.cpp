#include "lib/Utils/Utils.h"

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"   // from @llvm-project

namespace mlir {
namespace heir {

bool walkAndDetect(Operation *op, OpPredicate predicate) {
  return op
      ->walk([&](Operation *op) {
        return predicate(op) ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

LogicalResult validateValues(Operation *op, IsValidValueFn isValidValue) {
  bool argRes = llvm::all_of(op->getOperands(), [&](Value value) {
    return succeeded(isValidValue(value));
  });
  return !argRes ? failure() : success();
}

LogicalResult validateTypes(Operation *op, IsValidTypeFn isValidType) {
  bool argRes = llvm::all_of(op->getOperandTypes(), [&](Type type) {
    return succeeded(isValidType(type));
  });
  bool resultRes = llvm::all_of(op->getResultTypes(), [&](Type type) {
    return succeeded(isValidType(type));
  });
  return (!argRes || !resultRes) ? failure() : success();
}

LogicalResult walkAndValidateValues(Operation *op, IsValidValueFn isValidValue,
                                    std::optional<std::string> err) {
  LogicalResult res = success();
  op->walk([&](Operation *op) {
    res = validateValues(op, isValidValue);
    if (failed(res) && err.has_value()) op->emitError() << err.value();
    return failed(res) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return res;
}

}  // namespace heir
}  // namespace mlir
