#ifndef LIB_TARGET_UTIL_H_
#define LIB_TARGET_UTIL_H_

#include <functional>
#include <string>

#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Return a comma-separated string containing the values in the given
// ValueRange, with each value being converted to a string by the given mapping
// function.
std::string commaSeparatedValues(
    ValueRange values, std::function<std::string(Value)> valueToString);

// Return a comma-separated string containing the types in a given TypeRange,
// or failure if the mapper fails to convert any of the types.
FailureOr<std::string> commaSeparatedTypes(
    TypeRange types, std::function<FailureOr<std::string>(Type)> typeToString);

// Return a string containing the values in the given
// ValueRange enclosed in square brackets, with each value being converted to a
// string by the given mapping function, for example [1][2].
std::string bracketEnclosedValues(
    ValueRange values, std::function<std::string(Value)> valueToString);

// Returns a string expression for the flattened index of a MemRefType.
std::string flattenIndexExpression(
    MemRefType memRefType, ValueRange indices,
    std::function<std::string(Value)> valueToString);

// sum of products
std::string flattenIndexExpressionSOP(
    MemRefType memRefType, ValueRange indices,
    std::function<std::string(Value)> valueToString);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_UTIL_H_
