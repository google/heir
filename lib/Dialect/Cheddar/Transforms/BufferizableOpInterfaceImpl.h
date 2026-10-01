#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
