#ifndef LIB_DIALECT_RNS_IR_RNSOPS_H_
#define LIB_DIALECT_RNS_IR_RNSOPS_H_

#include <cstdint>

#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.h.inc"

namespace mlir {
namespace heir {
namespace rns {

template <typename Op>
LogicalResult verifyExtractSliceOp(Op* op, RNSType rnsType, int start,
                                   int size) {
  int64_t numLimbs = rnsType.getBasisTypes().size();

  if (start < 0) {
    return op->emitOpError()
           << "start index " << start << " cannot be negative";
  }

  if (size < 0) {
    return op->emitOpError() << "size " << size << " cannot be negative";
  }

  if (start + size > numLimbs) {
    return op->emitOpError()
           << "slice of size " << size << " starting at " << start
           << " is out of bounds for RNS type with " << numLimbs << " limbs";
  }

  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_RNS_IR_RNSOPS_H_
