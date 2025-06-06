#ifndef LIB_KERNEL_KERNEL_H_
#define LIB_KERNEL_KERNEL_H_

#include "mlir/include/mlir/IR/Operation.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project

namespace mlir {
namespace heir {

enum class KernelName {
  MatvecNaive,
  MatvecDiagonal,
};

bool isSupportedKernel(Operation *op, KernelName name);

}  // namespace heir
}  // namespace mlir
#endif  // LIB_KERNEL_KERNEL_H_
