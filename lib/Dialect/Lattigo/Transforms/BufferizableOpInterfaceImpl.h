#ifndef LIB_DIALECT_LATTIGO_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIB_DIALECT_LATTIGO_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LATTIGO_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
