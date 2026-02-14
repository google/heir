//===- CFAIRDialect.cpp - CFAIR dialect ---------------*- C++ -*-===//
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"

#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolDialect.cpp.inc"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/Support/CommandLine.h"     // from @llvm-project
#include "llvm/include/llvm/Support/InitLLVM.h"        // from @llvm-project
#include "llvm/include/llvm/Support/SourceMgr.h"       // from @llvm-project
#include "llvm/include/llvm/Support/ToolOutputFile.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"         // from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"        // from @llvm-project
#include "mlir/include/mlir/Support/FileUtilities.h"   // from @llvm-project
#include "mlir/include/mlir/Support/TypeID.h"          // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SCIFRBool dialect.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace scifrbool {

void SCIFRBoolDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolOps.cpp.inc"
      >();
}
}  // namespace scifrbool
}  // namespace mlir
