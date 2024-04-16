#ifndef INCLUDE_DIALECT_BGV_TRANSFORMS_ADDCLIENTINTERFACE_H_
#define INCLUDE_DIALECT_BGV_TRANSFORMS_ADDCLIENTINTERFACE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

#define GEN_PASS_DECL_ADDCLIENTINTERFACE
#include "include/Dialect/BGV/Transforms/Passes.h.inc"

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_BGV_TRANSFORMS_ADDCLIENTINTERFACE_H_
