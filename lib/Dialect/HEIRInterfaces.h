#ifndef LIB_DIALECT_HEIRINTERFACES_H_
#define LIB_DIALECT_HEIRINTERFACES_H_

#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

class ElementwiseByOperandOpInterface;

void registerOperandAndResultAttrInterface(DialectRegistry& registry);
void registerIncreasesMulDepthOpInterface(DialectRegistry& registry);
void registerPlaintextOperandInterface(DialectRegistry& registry);

LogicalResult verifyElementwiseByOperandImpl(
    ElementwiseByOperandOpInterface op);

// Folds a LimbwiseMappable operation for attributes that support
// LimbwiseAttrInterface. `op` must be an instance of
// LimbwiseMappableOpInterface, and the Attribute operands must be instances of
// LimbwiseAttrInterface.
//
// If the folding fails or the above preconditions fail, return {}
Attribute foldLimbwise(Operation* op, ArrayRef<Attribute> operands,
                       Type resultType);

}  // namespace heir
}  // namespace mlir

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRAttrInterfaces.h.inc"
#include "lib/Dialect/HEIROpInterfaces.h.inc"
#include "lib/Dialect/HEIRTypeInterfaces.h.inc"
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
// IWYU pragma: end_keep

#endif  // LIB_DIALECT_HEIRINTERFACES_H_
