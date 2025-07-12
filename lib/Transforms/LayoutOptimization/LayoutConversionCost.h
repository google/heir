#ifndef LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
#define LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_

#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

namespace mlir {
namespace heir {

using Cost = int64_t;
using tensor_ext::LayoutAttr;

Cost computeCostOfLayoutConversion(Value value, int64_t slots,
                                   LayoutAttr fromLayout, LayoutAttr toLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
