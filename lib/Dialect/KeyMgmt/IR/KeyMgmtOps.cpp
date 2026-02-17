#include "lib/Dialect/KeyMgmt/IR/KeyMgmtOps.h"

#include "lib/Dialect/KeyMgmt/IR/KeyMgmtDialect.h"
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtTypes.h"
#include "llvm/include/llvm/ADT/SetVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"     // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtOps.cpp.inc"
