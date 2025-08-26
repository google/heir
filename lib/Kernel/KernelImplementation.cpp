#include "lib/Kernel/KernelImplementation.h"

#include <cassert>
#include <memory>
#include <vector>

#include "lib/Kernel/KernelName.h"

namespace mlir {
namespace heir {
namespace kernel {

std::shared_ptr<KernelImplementation> implementKernel(KernelName kernelName,
                                                      ValueOrLiteral matrix,
                                                      ValueOrLiteral vector) {
  assert(kernelName == KernelName::MatvecDiagonal);
  auto matrixDag = KernelImplementation::leaf(matrix);
  auto vectorDag = KernelImplementation::leaf(vector);

  int numRows = matrix.shape[0];
  assert(numRows > 0);

  auto firstTerm =
      KernelImplementation::mul(KernelImplementation::leftRotate(vectorDag, 0),
                                KernelImplementation::extract(matrixDag, 0));

  auto accumulatedSum = firstTerm;
  for (int i = 1; i < numRows; ++i) {
    auto term = KernelImplementation::mul(
        KernelImplementation::leftRotate(vectorDag, i),
        KernelImplementation::extract(matrixDag, i));
    accumulatedSum = KernelImplementation::add(accumulatedSum, term);
  }
  return accumulatedSum;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
