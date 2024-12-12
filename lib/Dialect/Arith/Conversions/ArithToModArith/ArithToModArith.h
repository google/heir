#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTOMODARITH_H_
#define LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTOMODARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DECL
#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

}  // namespace arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTOMODARITH_H_
