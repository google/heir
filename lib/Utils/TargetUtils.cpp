#include "lib/Utils/TargetUtils.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>

#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

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

std::string flattenIndexExpression(ArrayRef<int64_t> shape,
                                   ArrayRef<std::string> indexStrings) {
  std::string accum = llvm::formatv("{0}", indexStrings[0]);
  for (size_t i = 1; i < indexStrings.size(); ++i) {
    accum =
        llvm::formatv("{0} + {1} * ({2})", indexStrings[i], shape[i], accum);
  }
  return accum;
}

std::string flattenIndexExpression(
    ShapedType type, ValueRange indices,
    std::function<std::string(Value)> valueToString) {
  return flattenIndexExpression(type.getShape(),
                                llvm::map_to_vector(indices, valueToString));
}

// sum of products
std::string flattenIndexExpressionSOP(
    MemRefType memRefType, ValueRange indices,
    std::function<std::string(Value)> valueToString) {
  const auto [strides, offset] = memRefType.getStridesAndOffset();
  std::string accum = std::to_string(offset);
  for (size_t i = 0; i < indices.size(); ++i) {
    accum = llvm::formatv("{2} + {0} * {1}", valueToString(indices[i]),
                          strides[i], accum);
  }

  return accum;
}

int64_t flattenedIndex(ShapedType type, ValueRange indices,
                       std::function<int64_t(Value)> valueToInt) {
  int index = valueToInt(indices[0]);
  for (size_t i = 1; i < indices.size(); i++) {
    index = valueToInt(indices[i]) + type.getShape()[i] * index;
  }
  return index;
}

int64_t flattenedIndex(MemRefType memRefType, ValueRange indices,
                       std::function<int64_t(Value)> valueToInt) {
  const auto [strides, offset] = memRefType.getStridesAndOffset();
  int index = offset;
  for (size_t i = 0; i < indices.size(); ++i) {
    index = index + strides[i] * valueToInt(indices[i]);
  }
  return index;
}

void emitFlattenedExtractSlice(ShapedType resultType, ShapedType sourceType,
                               StringRef resultName, StringRef sourceName,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> strides,
                               const OffsetToStringFn& offsetToString,
                               const InductionVarNameFn& getInductionVarName,
                               const LoopOpenerFn& loopOpener,
                               raw_indented_ostream& os) {
  // Emit a loop nest to copy the right values.
  // All tensors are flattened, so we need to keep track of the source flattened
  // index and the target flattened index.
  int64_t rank = offsets.size();
  SmallVector<std::string> inductionVarNames;
  for (int i = 0; i < rank; i++) {
    inductionVarNames.push_back(getInductionVarName(i));
  }

  // Create loop nest
  for (const auto& [i, inductionVar] : llvm::enumerate(inductionVarNames)) {
    // for (int64_t i0 = 0; i0 < size0; ++i0) {
    //
    // Note we are iterating over the target tensor's indices, so offsets and
    // strides must be accounted for when computing the source index.
    loopOpener(os, inductionVar, sizes[i]);
    os.indent();
  }

  os << resultName << "[";
  // Target index is just the flattened index of the result tensor. We use the
  // sizes, not the rank-reduced result type shape, since everything is
  // flattened and we just need to align things to the number of induction
  // variables.
  os << flattenIndexExpression(sizes, inductionVarNames);
  os << "] = " << sourceName << "[";
  // Source index requires incorporating the offsets and strides
  SmallVector<std::string> accessIndices;
  for (const auto& [i, inductionVar] : llvm::enumerate(inductionVarNames)) {
    accessIndices.push_back(llvm::formatv("{0} + {1} * {2}",
                                          offsetToString(offsets[i]),
                                          inductionVar, strides[i]));
  }
  os << flattenIndexExpression(sourceType.getShape(), accessIndices) << "];\n";

  // Close loop nest
  for (int i = 0; i < rank; i++) {
    os.unindent();
    os << "}\n";
  }
}

}  // namespace heir
}  // namespace mlir
