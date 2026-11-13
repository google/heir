#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir::heir::cheddar_gpu {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace mlir::heir::cheddar_gpu

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
