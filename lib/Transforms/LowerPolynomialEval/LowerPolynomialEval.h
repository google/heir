#ifndef LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_LOWERPOLYNOMIALEVAL_H_
#define LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_LOWERPOLYNOMIALEVAL_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LOWERPOLYNOMIALEVAL_LOWERPOLYNOMIALEVAL_H_
