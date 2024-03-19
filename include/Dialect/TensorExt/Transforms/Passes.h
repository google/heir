#ifndef INCLUDE_DIALECT_TENSOREXT_TRANSFORMS_PASSES_H_
#define INCLUDE_DIALECT_TENSOREXT_TRANSFORMS_PASSES_H_

#include "include/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "include/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"
#include "include/Dialect/TensorExt/Transforms/InsertRotate.h"
#include "include/Dialect/TensorExt/Transforms/RotateAndReduce.h"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_REGISTRATION
#include "include/Dialect/TensorExt/Transforms/Passes.h.inc"

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_TENSOREXT_TRANSFORMS_PASSES_H_
