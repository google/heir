#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_H_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Lattigo/IR/LattigoAttributes.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h.inc"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_H_
