#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_CHEDDARGPUTOGPU_H_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_CHEDDARGPUTOGPU_H_

#include <memory>

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DECL
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/CheddarGPUToGPU.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/CheddarGPUToGPU.h.inc"

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_CHEDDARGPUTOGPU_H_
