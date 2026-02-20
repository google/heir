#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POLYMULTONTT_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POLYMULTONTT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL_POLYMULTONTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createPolyMulToNTTPass();

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POLYMULTONTT_H_
