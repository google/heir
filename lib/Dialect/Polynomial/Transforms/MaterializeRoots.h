#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_MATERIALIZEROOTS_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_MATERIALIZEROOTS_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL_MATERIALIZEROOTS
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_MATERIALIZEROOTS_H_
