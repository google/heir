#ifndef LIB_TARGET_TFHERUST_UTILS_H_
#define LIB_TARGET_TFHERUST_UTILS_H_

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// Determine if the func can be emitted for tfhe-rs. If not, emit a
// warning and return success. This is because some functions are left
// over during compilation.
::mlir::LogicalResult canEmitFuncForTfheRust(::mlir::func::FuncOp &funcOp);

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUST_UTILS_H_
