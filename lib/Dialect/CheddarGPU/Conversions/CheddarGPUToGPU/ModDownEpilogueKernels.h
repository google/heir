#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_MODDOWNEPILOGUEKERNELS_H_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_MODDOWNEPILOGUEKERNELS_H_

#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

void populateModDownEpilogueToGPUPatterns(RewritePatternSet& patterns);
void configureModDownEpilogueGPUConversionTarget(ConversionTarget& target);

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_MODDOWNEPILOGUEKERNELS_H_
