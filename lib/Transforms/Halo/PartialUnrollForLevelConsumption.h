#ifndef LIB_TRANSFORMS_HALO_PARTIALUNROLLFORLEVELCONSUMPTION_H_
#define LIB_TRANSFORMS_HALO_PARTIALUNROLLFORLEVELCONSUMPTION_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DECL_PARTIALUNROLLFORLEVELCONSUMPTION
#include "lib/Transforms/Halo/Halo.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_HALO_PARTIALUNROLLFORLEVELCONSUMPTION_H_
