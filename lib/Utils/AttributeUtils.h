#ifndef LIB_UTILS_ATTRIBUTEUTILS_H_
#define LIB_UTILS_ATTRIBUTEUTILS_H_

#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

namespace mlir {
namespace heir {

// Find an attribute associated with the current value according to the
// following rules:
//
// 1. If the value is a block argument of a FunctionOpInterface, return the
//    argAttr associated to the function input.
// 2. If the value is a block argument of an op that implements
// OperandAndResultAttrInterface
//    (e.g., secret.generic and affine.for), find the operandAttr corresponding
//    to the operand associated with that block argument.
// 3. If the value is the result of an op that implements
// OperandAndResultAttrInterface,
//    find the resultAttr associated with that result.
// 4. Otherwise, find the attribute on the defining op.
FailureOr<Attribute> findAttributeAssociatedWith(Value value,
                                                 StringRef attrName);

void setAttributeAssociatedWith(Value value, StringRef attrName,
                                Attribute attr);

// Remove attributes with a given name from a given op, taking into account
// FunctionOpInterface's arg/result attrs as well as
// OperandAndResultAttrInterface.
void clearAttrs(Operation *op, StringRef attrName);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ATTRIBUTEUTILS_H_
