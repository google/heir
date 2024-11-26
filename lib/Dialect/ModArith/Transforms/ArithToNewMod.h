#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTONEWMOD_H_
#define LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTONEWMOD_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DECL
#include "lib/Dialect/ModArith/Transforms/ArithToNewMod.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/ModArith/Transforms/ArithToNewMod.h.inc"

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_ARITHTONEWMOD_H_
