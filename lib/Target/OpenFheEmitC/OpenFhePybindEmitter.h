#ifndef LIB_TARGET_OPENFHEEMITC_OPENFHEPYBINDEMITTER_H_
#define LIB_TARGET_OPENFHEEMITC_OPENFHEPYBINDEMITTER_H_

#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::openfhe {

LogicalResult translateToOpenFhePybind(Operation* op, llvm::raw_ostream& os);

}  // namespace mlir::heir::openfhe

#endif  // LIB_TARGET_OPENFHEEMITC_OPENFHEPYBINDEMITTER_H_
