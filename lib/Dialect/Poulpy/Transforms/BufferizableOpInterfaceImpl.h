#ifndef LIB_DIALECT_POULPY_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIB_DIALECT_POULPY_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poulpy {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POULPY_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
