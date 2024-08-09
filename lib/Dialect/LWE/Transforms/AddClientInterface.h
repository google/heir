#ifndef LIB_DIALECT_LWE_TRANSFORMS_ADDCLIENTINTERFACE_H_
#define LIB_DIALECT_LWE_TRANSFORMS_ADDCLIENTINTERFACE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DECL_ADDCLIENTINTERFACE
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_TRANSFORMS_ADDCLIENTINTERFACE_H_
