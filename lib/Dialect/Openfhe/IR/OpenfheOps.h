#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_H_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/Tablegen/InPlaceOpInterface.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h.inc"

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_H_
