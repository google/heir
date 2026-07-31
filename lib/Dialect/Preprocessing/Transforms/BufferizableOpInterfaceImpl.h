#ifndef LIB_DIALECT_PREPROCESSING_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIB_DIALECT_PREPROCESSING_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {

class DialectRegistry;

namespace heir {
namespace preprocessing {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
