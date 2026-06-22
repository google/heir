#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_UTIL_H_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_UTIL_H_

#include <cstdint>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

// Common type converter for preprocessing-to-* where the output is a
// single memref.
class SingleMemrefPreprocessingTypeConverter : public TypeConverter {
 public:
  SingleMemrefPreprocessingTypeConverter(
      const PreprocessingStorageLayoutAnalysis& analysis, Type targetType);

  const PreprocessingStorageLayoutAnalysis& getAnalysis() const {
    return analysis;
  }

  Type getTargetType() const { return targetType; }

  int64_t getFlatBaseOffset(PreprocessingStorageType storageTy,
                            Type elementType, uint32_t siteId) const;

 private:
  const PreprocessingStorageLayoutAnalysis& analysis;
  Type targetType;
};

// Populate common memref conversion patterns for single-typed preprocessing
// storage.
void populateCommonPreprocessingToMemrefPatterns(
    const SingleMemrefPreprocessingTypeConverter& typeConverter,
    RewritePatternSet& patterns);

// Populate patterns for LWETo* passes that need to convert preprocessing ops.
// Note a special pattern is needed (instead of ConvertAny) because the type
// attribute on preprocessing.store/load must be converted manually.
void populatePreprocessingConversions(RewritePatternSet& patterns,
                                      const TypeConverter& typeConverter,
                                      MLIRContext* context);

// Preprocessing conversions lower a (globally unique) preprocessing.storage SSA
// value to one or more flat memrefs (one for each type).
//
// getLinearIndex takes as input the data about a particular preprocessing.store
// or preprocessing.load op, and outputs an SSA value corresponding to the index
// within the flat memref of that value.
//
// The PreprocessingStorageLayoutAnalysis calculates offsets for each site_id,
// which corresponds to a target encode op in the original IR. Given this base
// offset, the variadic indices are matched with induction variables of an
// enclosing loop nest, and flattened into an SSA value corresponding to a 1D
// absolute index. If there is no containing loop, nest the relative offset is
// 0. Otherwise it is a flattened (start, stop, step)-appropriate shift from
// the base offset for the given site_id.
FailureOr<Value> getLinearIndex(OpBuilder& builder, Location loc, Operation* op,
                                int64_t baseOffset, ValueRange indices);

// Helpers for LWETo* that have to convert preprocessing ops and types.
// At this stage, the preprocessing.storage may have multiple element types,
// so the type conversion is needed to combine them into a single type when
// the target backend has a single type, such as openfhe.plaintext.
inline Type convertStorageElementTypes(PreprocessingStorageType storageTy,
                                       TypeConverter* typeConverter) {
  SmallVector<Type> convertedTypes;
  for (Type t : storageTy.getElementTypes()) {
    convertedTypes.push_back(typeConverter->convertType(t));
  }
  return preprocessing::PreprocessingStorageType::get(storageTy.getContext(),
                                                      convertedTypes);
}

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_UTIL_H_
