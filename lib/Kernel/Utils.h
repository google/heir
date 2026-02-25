#ifndef LIB_KERNEL_UTILS_H_
#define LIB_KERNEL_UTILS_H_

#include "lib/Kernel/ArithmeticDag.h"
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

Type dagTypeToMLIRType(const DagType& dagType, OpBuilder& builder);

DagType mlirTypeToDagType(Type type);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_UTILS_H_
