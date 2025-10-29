#ifndef LIB_KERNEL_KERNELNAME_H_
#define LIB_KERNEL_KERNELNAME_H_

#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {

enum KernelName : int {
  // A trivial kernel is one for which there is only a single known option
  // (e.g., an elementwise addition).
  Trivial = 0,

  // TODO(#1589): implement MatvecNaive kernel
  MatvecNaive,

  // Plaintext-ciphertext matvec using the Halevi-Shoup diagonal method.
  MatvecDiagonal,

  // Ciphertext-plaintext vecmat using the Halevi-Shoup diagonal method
  // (transpose of MatvecDiagonal).
  VecmatDiagonal,

  // LHS matrix is secret, RHS is a plaintext matrix. This expands the matmul
  // into a single matvec using the diagonal matrix vector kernel.
  MatmulDiagonal,

  // Ciphertext-ciphertext matmul using the bicyclic packing method.
  MatmulBicyclic,
};

std::ostream& operator<<(std::ostream& os, const KernelName& k);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const KernelName& k);
mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const KernelName& k);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_KERNELNAME_H_
