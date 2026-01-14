#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_ALLOCTOINPLACE_H_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_ALLOCTOINPLACE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DECL_ALLOCTOINPLACE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_ALLOCTOINPLACE_H_
