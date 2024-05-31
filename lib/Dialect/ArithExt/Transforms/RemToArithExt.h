#ifndef LIB_DIALECT_ARITHEXT_TRANSFORMS_REMTOARITHEXT_H_
#define LIB_DIALECT_ARITHEXT_TRANSFORMS_REMTOARITHEXT_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace arith_ext {

#define GEN_PASS_DECL_REMTOARITHEXT
#include "lib/Dialect/ArithExt/Transforms/Passes.h.inc"

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ARITHEXT_TRANSFORMS_REMTOARITHEXT_H_
