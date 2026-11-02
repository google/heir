#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_HOISTKERNELS_H_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_HOISTKERNELS_H_

#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

void populateHoistToGPUPatterns(RewritePatternSet& patterns);
void configureHoistGPUConversionTarget(ConversionTarget& target);

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_HOISTKERNELS_H_
