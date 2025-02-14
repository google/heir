#ifndef LIB_DIALECT_MODARITH_IR_MODARITHOPS_H_
#define LIB_DIALECT_MODARITH_IR_MODARITHOPS_H_

// NOLINTBEGIN(misc-include-cleaner): Required to define ModArithOps
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.h.inc"

namespace mlir {
namespace heir {
namespace mod_arith {

template <typename OpType>
inline ModArithType getResultModArithType(OpType op) {
  return cast<ModArithType>(getElementTypeOrSelf(op.getResult().getType()));
}

template <typename OpType>
inline ModArithType getOperandModArithType(OpType op) {
  return cast<ModArithType>(getElementTypeOrSelf(op.getOperand().getType()));
}

template <typename OpType>
inline IntegerType getResultIntegerType(OpType op) {
  return cast<IntegerType>(getElementTypeOrSelf(op.getResult().getType()));
}

template <typename OpType>
inline IntegerType getOperandIntegerType(OpType op) {
  return cast<IntegerType>(getElementTypeOrSelf(op.getOperand().getType()));
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHOPS_H_
