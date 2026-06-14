#ifndef LIB_TARGET_OPENFHEEMITC_OPENFHEEMITCEMITTER_H_
#define LIB_TARGET_OPENFHEEMITC_OPENFHEEMITCEMITTER_H_

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

namespace mlir {
class Operation;
}  // namespace mlir

namespace mlir::heir::openfhe {

/// Translates the given operation to OpenFHE C++ code using the EmitC dialect.
LogicalResult translateToOpenFheEmitC(Operation* op, llvm::raw_ostream& os);

}  // namespace mlir::heir::openfhe

#endif  // LIB_TARGET_OPENFHEEMITC_OPENFHEEMITCEMITTER_H_
