#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_

#include <cstdint>

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir::heir::cheddar {

// `GetRotKeyOp` stores a static distance attribute, but dynamic
// `ckks.rotate` lowering still needs a placeholder key op so the emitter can
// trace back to the `UserInterface`. This sentinel distance marks that case.
constexpr int64_t kDynamicRotationKeyDistanceSentinel = -1;

}  // namespace mlir::heir::cheddar

#define GET_OP_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarOps.h.inc"

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
