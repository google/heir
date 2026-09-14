#ifndef LIB_TRANSFORMS_LOWERINGHISTORY_LOWERINGHISTORY_H_
#define LIB_TRANSFORMS_LOWERINGHISTORY_LOWERINGHISTORY_H_

#include "mlir/include/mlir/IR/Location.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"    // from @llvm-project

namespace mlir::heir {
// Record an explicit successful lowering when the source carries history.
// Otherwise preserve the original location without adding metadata.
Location getLoweringLocation(Operation* source, StringRef pass,
                             StringRef resultOperation);

#define GEN_PASS_DECL
#include "lib/Transforms/LoweringHistory/LoweringHistory.h.inc"
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LoweringHistory/LoweringHistory.h.inc"
}  // namespace mlir::heir

#endif  // LIB_TRANSFORMS_LOWERINGHISTORY_LOWERINGHISTORY_H_
