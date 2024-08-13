#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_H_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_H_

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DECL_POLYMULTONTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_H_
