#ifndef LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_RELINEARIZE_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_RELINEARIZE_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DECL_DECOMPOSERELINEARIZE
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_RELINEARIZE_H_
