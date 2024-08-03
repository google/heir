#ifndef LIB_TRANSFORMS_TENSORTOSCALARS_TENSORTOSCALARS_H_
#define LIB_TRANSFORMS_TENSORTOSCALARS_TENSORTOSCALARS_H_
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/TensorToScalars/TensorToScalars.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/TensorToScalars/TensorToScalars.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_TENSORTOSCALARS_TENSORTOSCALARS_H_
