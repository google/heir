#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_VALUEUTILS_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_VALUEUTILS_H_

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project

namespace mlir::heir::rotom {

// Unwraps a secret.secret type to the value type it protects; returns `type`
// unchanged for non-secret types.
Type getPlainValueType(Type type);

// Returns true if `value` is (a secret of) a ranked tensor -- the values the
// layout assignment reasons about.
bool isTensorLike(Value value);

// Returns true if `layout` can describe `value`'s tensor type: every
// non-gap/non-replicate dim indexes a real tensor dimension and its
// size*stride footprint fits the padded extent of that dimension.
bool isLayoutCompatibleWithValue(LayoutAttr layout, Value value);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_VALUEUTILS_H_
