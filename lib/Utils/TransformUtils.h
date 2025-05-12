#ifndef LIB_UTILS_TRANSFORMUTILS_H_
#define LIB_UTILS_TRANSFORMUTILS_H_

#include <string_view>

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project

namespace mlir {
namespace heir {

func::FuncOp detectEntryFunction(ModuleOp moduleOp,
                                 std::string_view entryFunction);

/// Replace an integer-typed value with a tensor of the individual bits. The
/// lowest order bit of the integer is the first element in the tensor.
Value convertIntegerValueToTensorOfBits(Value integer, OpBuilder &b,
                                        Location loc);

/// Replace a 1D tensor of bits with an integer-typed value. The bits are
/// interpreted so that the first element of the tensor is the lowest order bit
/// of the result integer.
Value convertTensorOfBitsToInteger(Value tensor, Type resultType, OpBuilder &b,
                                   Location loc);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_TRANSFORMUTILS_H_
