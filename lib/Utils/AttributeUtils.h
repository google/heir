#ifndef LIB_UTILS_ATTRIBUTEUTILS_H_
#define LIB_UTILS_ATTRIBUTEUTILS_H_

#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project

namespace mlir {
namespace heir {

Attribute getAttributeFromValue(Value value, StringRef attrName);
void setAttributeForValue(Value value, StringRef attrName, Attribute attr);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ATTRIBUTEUTILS_H_
