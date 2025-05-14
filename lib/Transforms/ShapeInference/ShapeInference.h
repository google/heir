#ifndef LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_H_
#define LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/ShapeInference/ShapeInference.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/ShapeInference/ShapeInference.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_H_
