#ifndef LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
#define LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_

#include <cstdint>

#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"       // from @llvm-project

namespace mlir {
namespace heir {

using Cost = int64_t;

Cost computeCostOfLayoutConversion(Value value, int64_t slots,
                                   Attribute fromLayout, Attribute toLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
