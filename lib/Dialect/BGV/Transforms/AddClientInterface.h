#ifndef LIB_DIALECT_BGV_TRANSFORMS_ADDCLIENTINTERFACE_H_
#define LIB_DIALECT_BGV_TRANSFORMS_ADDCLIENTINTERFACE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

#define GEN_PASS_DECL_ADDCLIENTINTERFACE
#include "lib/Dialect/BGV/Transforms/Passes.h.inc"

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_BGV_TRANSFORMS_ADDCLIENTINTERFACE_H_
