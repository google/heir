#ifndef LIB_UTILS_TARGETUTILS_H_
#define LIB_UTILS_TARGETUTILS_H_

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>

#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

// Return a comma-separated string containing the values in the given
// ValueRange, with each value being converted to a string by the given mapping
// function.
std::string commaSeparatedValues(
    ValueRange values, std::function<std::string(Value)> valueToString);

// Return a comma-separated string containing the values in the given
// ArrayRef, with each value being converted to a string by std::to_string
template <typename T>
std::string commaSeparated(ArrayRef<T> values) {
  if (values.empty()) {
    return std::string();
  }
  return std::accumulate(
      std::next(values.begin()), values.end(), std::to_string(values[0]),
      [&](const std::string& a, T b) { return a + ", " + std::to_string(b); });
}

// Return a comma-separated string containing the types in a given TypeRange,
// or failure if the mapper fails to convert any of the types.
FailureOr<std::string> commaSeparatedTypes(
    TypeRange types, std::function<FailureOr<std::string>(Type)> typeToString);

// Return a string containing the values in the given
// ValueRange enclosed in square brackets, with each value being converted to a
// string by the given mapping function, for example [1][2].
std::string bracketEnclosedValues(
    ValueRange values, std::function<std::string(Value)> valueToString);

// Returns a string expression for the flattened index of a ShapedType.
std::string flattenIndexExpression(
    ShapedType type, ValueRange indices,
    std::function<std::string(Value)> valueToString);

// sum of products
std::string flattenIndexExpressionSOP(
    MemRefType memRefType, ValueRange indices,
    std::function<std::string(Value)> valueToString);

int64_t flattenedIndex(ShapedType type, ValueRange indices,
                       std::function<int64_t(Value)> valueToInt);

int64_t flattenedIndex(MemRefType memRefType, ValueRange indices,
                       std::function<int64_t(Value)> valueToInt);

inline bool isDebugPort(StringRef debugPortName) {
  return debugPortName.rfind("__heir_debug") == 0;
}

inline StringRef canonicalizeDebugPort(StringRef debugPortName) {
  if (isDebugPort(debugPortName)) {
    return "__heir_debug";
  }
  return debugPortName;
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_TARGETUTILS_H_
