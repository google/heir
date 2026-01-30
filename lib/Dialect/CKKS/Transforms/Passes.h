#ifndef LIB_DIALECT_CKKS_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_PASSES_H_

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/Transforms/DecomposeRelinearize.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_PASSES_H_
