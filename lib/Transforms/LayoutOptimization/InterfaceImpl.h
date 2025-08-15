#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_

#include "lib/Transforms/LayoutOptimization/Hoisting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project

namespace mlir {
namespace heir {

/// Construct a trivial hoister for which all layouts can be hoisted without
/// any kernel change or difference in the output layout.
Hoister createTrivialHoister(Operation* op);

/// Construct a hoister that pre-composes the slot dimension of the Matvec op's
/// matrix with the incremental transformation required to go from vecFromLayout
/// to vecToLayout, while keeping the kernel the same.
Hoister createPrecomposingMatvecHoister(linalg::MatvecOp op);

void registerLayoutConversionHoistableInterface(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
