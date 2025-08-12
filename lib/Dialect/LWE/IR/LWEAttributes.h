#ifndef LIB_DIALECT_LWE_IR_LWEATTRIBUTES_H_
#define LIB_DIALECT_LWE_IR_LWEATTRIBUTES_H_

#include <cstdint>

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEEnums.h.inc"
#include "mlir/include/mlir/IR/Attributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"     // from @llvm-project
#include "mlir/include/mlir/IR/TensorEncoding.h"  // from @llvm-project
// Required to pull in poly's Ring_Attr
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWEAttributes.h.inc"

namespace mlir {
namespace heir {
namespace lwe {

int64_t getScalingFactorFromEncodingAttr(Attribute encoding);

PlaintextSpaceAttr inferMulOpPlaintextSpaceAttr(MLIRContext* ctx,
                                                PlaintextSpaceAttr x,
                                                PlaintextSpaceAttr y);

PlaintextSpaceAttr inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
    MLIRContext* ctx, PlaintextSpaceAttr x, APInt dividedModulus);

Attribute getEncodingAttrWithNewScalingFactor(Attribute encoding,
                                              int64_t newScale);

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_IR_LWEATTRIBUTES_H_
