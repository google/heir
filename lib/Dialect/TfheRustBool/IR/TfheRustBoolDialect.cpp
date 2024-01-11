#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"

#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.cpp.inc"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.cpp.inc"

namespace mlir {
namespace heir {
namespace tfhe_rust_bool {

void TfheRustBoolDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.cpp.inc"
      >();
}

}  // namespace tfhe_rust_bool
}  // namespace heir
}  // namespace mlir
