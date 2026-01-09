#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_H_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/Tablegen/InplaceOpInterface.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Lattigo/IR/LattigoOps.h.inc"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_H_
