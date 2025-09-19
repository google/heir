#ifndef LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
#define LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_

#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

namespace mlir {
namespace heir {

int64_t computeCostOfLayoutConversion(int64_t numCiphertexts,
                                      int64_t ciphertextSize,
                                      tensor_ext::NewLayoutAttr fromLayout,
                                      tensor_ext::NewLayoutAttr toLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
