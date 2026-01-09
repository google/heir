#ifndef LIB_DIALECT_MGMT_IR_MGMTOPS_H_
#define LIB_DIALECT_MGMT_IR_MGMTOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"   // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Mgmt/IR/MgmtOps.h.inc"

namespace mlir {
namespace heir {
namespace mgmt {

/// Remove all unused mgmt.init ops from the top operation.
void cleanupInitOp(Operation* top);

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_IR_MGMTOPS_H_
