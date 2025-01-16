#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_H_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_H_

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h.inc"

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_H_
