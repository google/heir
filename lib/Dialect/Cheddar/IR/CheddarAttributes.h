#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_

#include <optional>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h.inc"

namespace mlir {
namespace heir {
namespace cheddar {

constexpr ::llvm::StringLiteral kParameterSetAttrName = "cheddar.parameter_set";

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_
