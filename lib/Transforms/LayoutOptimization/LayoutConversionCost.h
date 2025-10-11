#ifndef LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
#define LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_

#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

namespace mlir {
namespace heir {

using Cost = int64_t;

Cost computeCostOfLayoutConversion(int64_t numCiphertexts,
                                   int64_t ciphertextSize,
                                   tensor_ext::LayoutAttr fromLayout,
                                   tensor_ext::LayoutAttr toLayout,
                                   std::size_t vveRandomSeed,
                                   unsigned vveRandomTries);

Cost computeCostOfLayoutConversion(int64_t ciphertextSize, Attribute fromLayout,
                                   Attribute toLayout,
                                   std::size_t vveRandomSeed,
                                   unsigned vveRandomTries);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORM_LAYOUTCONVERSION_LAYOUTCONVERSION_H_
