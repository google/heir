#ifndef LIB_TRANSFORMS_HALO_PASSES_H_
#define LIB_TRANSFORMS_HALO_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/Halo/ReconcileMixedSecretnessIterArgs.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"           // @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/Halo/Halo.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_HALO_PASSES_H_
