#ifndef LIB_DIALECT_MGMT_TRANSFORMS_ANNOTATEMGMT_H_
#define LIB_DIALECT_MGMT_TRANSFORMS_ANNOTATEMGMT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

#define GEN_PASS_DECL_ANNOTATEMGMT
#include "lib/Dialect/Mgmt/Transforms/Passes.h.inc"

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_TRANSFORMS_ANNOTATEMGMT_H_
