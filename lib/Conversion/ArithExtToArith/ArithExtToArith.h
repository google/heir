#ifndef LIB_DIALECT_ARITHEXT_TRANSFORMS_ARITHEXTTOARITH_H_
#define LIB_DIALECT_ARITHEXT_TRANSFORMS_ARITHEXTTOARITH_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace arith_ext {

#define GEN_PASS_DECL
#include "lib/Conversion/ArithExtToArith/ArithExtToArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/ArithExtToArith/ArithExtToArith.h.inc"

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ARITHEXT_TRANSFORMS_ARITHEXTTOARITH_H_
