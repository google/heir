#ifndef LIB_DIALECT_HEIRINTERFACES_H_
#define LIB_DIALECT_HEIRINTERFACES_H_

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

namespace mlir {
namespace heir {
// Pull in HEIR interfaces
#include "lib/Dialect/HEIRInterfaces.h.inc"

SmallVector<int32_t> findAllRelinBasis(func::FuncOp op);

SmallVector<int64_t> findAllRotIndices(func::FuncOp op);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_HEIRINTERFACES_H_
