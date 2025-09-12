#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_INTERFACEIMPL_H_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_INTERFACEIMPL_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {

void registerOperandLayoutRequirementOpInterface(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_INTERFACEIMPL_H_
