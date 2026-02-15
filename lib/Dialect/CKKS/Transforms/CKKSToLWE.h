#ifndef LIB_DIALECT_CKKS_TRANSFORMS_CKKSTOLWE_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_CKKSTOLWE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DECL_CKKSTOLWE
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_CKKSTOLWE_H_
