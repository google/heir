#ifndef LIB_TRANSFORMS_SECRETWITHMGMT_PASSES_H_
#define LIB_TRANSFORMS_SECRETWITHMGMT_PASSES_H_

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/SecretWithMgmt/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/SecretWithMgmt/Passes.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETWITHMGMT_PASSES_H_
