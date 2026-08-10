#ifndef LIB_DIALECT_REDUCESLEVELOPINTERFACEREGISTRATION_H_
#define LIB_DIALECT_REDUCESLEVELOPINTERFACEREGISTRATION_H_

namespace mlir {
class DialectRegistry;

namespace heir {

void registerReducesLevelOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_REDUCESLEVELOPINTERFACEREGISTRATION_H_
