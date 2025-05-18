#ifndef LIB_TRANSFORMS_GENERATEPARAM_UTILS_H_
#define LIB_TRANSFORMS_GENERATEPARAM_UTILS_H_

#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult copyMgmtAttrToClientHelpers(Operation *op);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_GENERATEPARAM_UTILS_H_
