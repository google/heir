#ifndef LIB_TRANSFORMS_PREPAREFORLAYOUTPROPAGATION_PREPAREFORLAYOUTPROPAGATION_H_
#define LIB_TRANSFORMS_PREPAREFORLAYOUTPROPAGATION_PREPAREFORLAYOUTPROPAGATION_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/PrepareForLayoutPropagation/PrepareForLayoutPropagation.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/PrepareForLayoutPropagation/PrepareForLayoutPropagation.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_PREPAREFORLAYOUTPROPAGATION_PREPAREFORLAYOUTPROPAGATION_H_
