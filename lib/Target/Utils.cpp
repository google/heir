#include "lib/Target/Utils.h"

#include <functional>
#include <iterator>
#include <numeric>
#include <string>

#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

namespace mlir {
namespace heir {

std::string commaSeparatedValues(
    ValueRange values, std::function<std::string(Value)> valueToString) {
  if (values.empty()) {
    return std::string();
  }
  return std::accumulate(std::next(values.begin()), values.end(),
                         valueToString(values[0]),
                         [&](const std::string& a, Value b) {
                           return a + ", " + valueToString(b);
                         });
}

FailureOr<std::string> commaSeparatedTypes(
    TypeRange types, std::function<FailureOr<std::string>(Type)> typeToString) {
  if (types.empty()) {
    return std::string();
  }
  return std::accumulate(
      std::next(types.begin()), types.end(), typeToString(types[0]),
      [&](FailureOr<std::string> a, Type b) -> FailureOr<std::string> {
        auto result = typeToString(b);
        if (failed(result)) {
          return failure();
        }
        return a.value() + ", " + result.value();
      });
}

std::string bracketEnclosedValues(
    ValueRange values, std::function<std::string(Value)> valueToString) {
  if (values.empty()) {
    return std::string();
  }
  return std::accumulate(std::next(values.begin()), values.end(),
                         "[" + valueToString(values[0]) + "]",
                         [&](const std::string& a, Value b) {
                           return a + "[" + valueToString(b) + "]";
                         });
}

std::string flattenIndexExpression(
    MemRefType memRefType, ValueRange indices,
    std::function<std::string(Value)> valueToString) {
  std::string accum = llvm::formatv("{0}", valueToString(indices[0]));
  for (size_t i = 1; i < indices.size(); ++i) {
    accum = llvm::formatv("{0} + {1} * ({2})", valueToString(indices[i]),
                          memRefType.getShape()[i], accum);
  }
  return accum;
}

// sum of products
std::string flattenIndexExpressionSOP(
    MemRefType memRefType, ValueRange indices,
    std::function<std::string(Value)> valueToString) {
  const auto [strides, offset] = getStridesAndOffset(memRefType);
  std::string accum = std::to_string(offset);
  for (size_t i = 0; i < indices.size(); ++i) {
    accum = llvm::formatv("{2} + {0} * {1}", valueToString(indices[i]),
                          strides[i], accum);
  }

  return accum;
}

}  // namespace heir
}  // namespace mlir
