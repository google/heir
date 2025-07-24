#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/TensorEncoding.h"    // from @llvm-project
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h.inc"

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_
