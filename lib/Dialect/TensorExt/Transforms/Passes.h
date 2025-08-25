#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/Transforms/CollapseInsertionChains.h"
#include "lib/Dialect/TensorExt/Transforms/FoldConvertLayoutIntoAssignLayout.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Dialect/TensorExt/Transforms/InsertRotate.h"
#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"

// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_H_
