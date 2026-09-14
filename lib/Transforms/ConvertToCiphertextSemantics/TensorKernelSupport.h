#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TENSORKERNELSUPPORT_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TENSORKERNELSUPPORT_H_

#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir {
// Preconditions shared by the early validator and ciphertext conversion.
// Call only when the source tensor is secret; cleartext kernels are
// unrestricted.
LogicalResult checkEncryptedExtract(tensor::ExtractOp op);
LogicalResult checkEncryptedPad(tensor::PadOp op);
}  // namespace mlir::heir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_TENSORKERNELSUPPORT_H_
