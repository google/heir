#ifndef LIB_DIALECT_MGMT_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_MGMT_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"

namespace mlir {
namespace heir {
namespace mgmt {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Mgmt/Transforms/Passes.h.inc"

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_TRANSFORMS_PASSES_H_
