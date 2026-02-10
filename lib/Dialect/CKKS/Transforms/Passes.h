#ifndef LIB_DIALECT_CKKS_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_PASSES_H_

#include "lib/Dialect/CKKS/Transforms/DecomposeKeySwitch.h"
#include "lib/Dialect/CKKS/Transforms/DecomposeRelinearize.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_PASSES_H_
