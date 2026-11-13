#ifndef LIB_DIALECT_CHEDDARGPU_UTILS_CONSTANTTABLES_H_
#define LIB_DIALECT_CHEDDARGPU_UTILS_CONSTANTTABLES_H_

#include <cstdint>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

/// Constructs a DenseElementsAttr containing all primes encoded by `params`
/// in Cheddar limb order: ter[num_ter - 1 ... 0], main[0 ... num_main - 1],
/// aux[0 ... num_aux - 1].
DenseElementsAttr getDevicePrimesConstant(MLIRContext* ctx, ParamsAttr params,
                                          uint32_t word);

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_UTILS_CONSTANTTABLES_H_
