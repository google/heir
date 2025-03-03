#ifndef LIB_UTILS_TRANSFORMUTILS_H_
#define LIB_UTILS_TRANSFORMUTILS_H_

#include <string_view>

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project

namespace mlir {
namespace heir {

func::FuncOp detectEntryFunction(ModuleOp moduleOp,
                                 std::string_view entryFunction);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_TRANSFORMUTILS_H_
