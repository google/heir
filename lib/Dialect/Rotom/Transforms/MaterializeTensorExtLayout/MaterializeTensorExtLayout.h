#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_MATERIALIZETENSOREXTLAYOUT_MATERIALIZETENSOREXTLAYOUT_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_MATERIALIZETENSOREXTLAYOUT_MATERIALIZETENSOREXTLAYOUT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

#define GEN_PASS_DECL_MATERIALIZETENSOREXTLAYOUT
#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h.inc"

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_MATERIALIZETENSOREXTLAYOUT_MATERIALIZETENSOREXTLAYOUT_H_
