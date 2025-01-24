#ifndef LIB_UTILS_UTILS_H_
#define LIB_UTILS_UTILS_H_

#include <functional>
#include <optional>
#include <string>

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

typedef std::function<bool(Operation *)> OpPredicate;
typedef std::function<LogicalResult(const Type &)> IsValidTypeFn;
typedef std::function<LogicalResult(const Value &)> IsValidValueFn;

typedef std::function<bool(const Type &)> TypePredicate;

typedef std::function<bool(Dialect *)> DialectPredicate;

template <typename... OpTys>
OpPredicate OpEqual() {
  return [](Operation *op) { return mlir::isa<OpTys...>(op); };
}

template <typename... TypeTys>
TypePredicate TypeEqual() {
  return [](const Type &type) { return mlir::isa<TypeTys...>(type); };
}

template <typename... DialectTys>
DialectPredicate DialectEqual() {
  return [](Dialect *dialect) { return mlir::isa<DialectTys...>(dialect); };
}

// Walks the given op, applying the predicate to traversed ops until the
// predicate returns true, then returns the operation that matched, or
// nullptr if there were no matches.
Operation *walkAndDetect(Operation *op, OpPredicate predicate);

// specialization for detecting a specific operation type
template <typename... OpTys>
bool containsAnyOperations(Operation *op) {
  Operation *foundOp = walkAndDetect(op, OpEqual<OpTys...>());
  return foundOp != nullptr;
}

/// Apply isValidType to the operands and results, returning an appropriate
/// logical result.
LogicalResult validateTypes(Operation *op, IsValidTypeFn isValidType);

/// Apply isValidType to the operands, returning an appropriate logical result.
LogicalResult validateValues(Operation *op, IsValidValueFn isValidValue);

/// Walk the IR and apply a predicate to all argument values
/// encountered, returning failure if any type is invalid. Invalidity is
/// determined by whether the IsValidTypeFn returns a failed LogicalResult. If
/// err is provided, report and error to the user when the first invalid value
/// is encountered.
LogicalResult walkAndValidateValues(
    Operation *op, IsValidValueFn isValidValue,
    std::optional<std::string> err = std::nullopt);

/// Walk the IR and apply a predicate to all argument and result types
/// encountered, returning failure if any type is invalid. Invalidity is
/// determined by whether the IsValidTypeFn returns a failed LogicalResult.
/// If err is provided, report and error to the user when the first invalid type
/// is encountered.
template <typename OpTy>
LogicalResult walkAndValidateTypes(
    Operation *op, IsValidTypeFn isValidType,
    std::optional<std::string> err = std::nullopt) {
  LogicalResult res = success();
  op->walk([&](OpTy op) {
    res = validateTypes(op.getOperation(), isValidType);
    if (failed(res) && err.has_value()) op->emitError() << err.value();
    return failed(res) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return res;
}

// Returns true if the op contains ops from the given dialects.
template <typename... Dialects>
bool containsDialects(Operation *op) {
  Operation *foundOp = walkAndDetect(op, [&](Operation *op) {
    return DialectEqual<Dialects...>()(op->getDialect());
  });
  return foundOp != nullptr;
}

// Returns true if the op contains argument values of the given type.
// NOTE: any_of instead of all_of
bool containsArgumentOfType(Operation *op, TypePredicate predicate);

template <typename... TypeTys>
bool containsArgumentOfType(Operation *op) {
  return containsArgumentOfType(op, TypeEqual<TypeTys...>());
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_UTILS_H_