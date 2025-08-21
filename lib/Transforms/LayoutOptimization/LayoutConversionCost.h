#ifndef LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
#define LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_

#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

namespace mlir {
namespace heir {

using Cost = int64_t;
using tensor_ext::LayoutAttr;

// TODO(#2047) migrate to relation-based layout attr
Cost computeCostOfLayoutConversion(Value value, int64_t slots,
                                   LayoutAttr fromLayout, LayoutAttr toLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
