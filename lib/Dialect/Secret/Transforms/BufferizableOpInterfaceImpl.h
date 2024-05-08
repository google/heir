#ifndef HEIR_LIB_DIALECT_SECRET_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define HEIR_LIB_DIALECT_SECRET_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {

class DialectRegistry;

namespace heir {
namespace secret {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // HEIR_LIB_DIALECT_SECRET_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
