#ifndef LIB_KERNEL_UTILS_H_
#define LIB_KERNEL_UTILS_H_

#include "lib/Kernel/ArithmeticDag.h"
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

Type dagTypeToMLIRType(const DagType& dagType, OpBuilder& builder);

/// Convert an mlir type to an appropriate DagType. For some input types, like a
/// Lattigo CKKS ciphertext type, the number of slots is not encoded on the type
/// alone, and extra data should be passed in to say how many slots should be
/// used.
DagType mlirTypeToDagType(Type type, int numSlots = 8192);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_UTILS_H_
