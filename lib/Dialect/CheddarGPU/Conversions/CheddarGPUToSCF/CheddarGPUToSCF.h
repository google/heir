#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOSCF_CHEDDARGPUTOSCF_H_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOSCF_CHEDDARGPUTOSCF_H_

#include <memory>

#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

void populateBConvToSCFPatterns(RewritePatternSet& patterns);
void populateNTTToSCFPatterns(RewritePatternSet& patterns);
void populateModDownEpilogueToSCFPatterns(RewritePatternSet& patterns);

#define GEN_PASS_DECL
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/CheddarGPUToSCF.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/CheddarGPUToSCF.h.inc"

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOSCF_CHEDDARGPUTOSCF_H_
