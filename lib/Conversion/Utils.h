#ifndef LIB_CONVERSION_UTILS_H_
#define LIB_CONVERSION_UTILS_H_

#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

struct ConvertAny : public ConversionPattern {
  ConvertAny(const TypeConverter &anyTypeConverter, MLIRContext *context);

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

// Adds conversion patterns that deal with tensor<..xsource_type>
// when source_type will be type converted to tensor<...>, too
void addTensorOfTensorConversionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);

// Adds the standard set of conversion patterns for
// converting types involved in func, cf, etc., which
// don't depend on the logic of the dialect beyond the
// type converter.
void addStructuralConversionPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target);

// Seems like this would be better as a method on the
// LWE_EncodingAttrWithScalingFactor class, but I still have the problem of the
// type returned by getEncoding being a vanilla Attribute. Probably we need a
// common interface for LWE_EncodingAttrWithScalingFactor, and cast to that?
int widthFromEncodingAttr(Attribute encoding);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_CONVERSION_UTILS_H_
