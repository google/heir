#ifndef LIB_KERNEL_KERNELNAME_H_
#define LIB_KERNEL_KERNELNAME_H_

#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

namespace mlir {
namespace heir {

enum class KernelName {
  // A trivial kernel is one for which there is only a single known option
  // (e.g., an elementwise addition).
  Trivial,
  MatvecNaive,
  MatvecDiagonal,
};

}  // namespace heir

std::ostream& operator<<(std::ostream& os, const heir::KernelName& k);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const heir::KernelName& k);
mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const heir::KernelName& k);

}  // namespace mlir

#endif  // LIB_KERNEL_KERNELNAME_H_
