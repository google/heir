//===- CFAIRDialect.cpp - CFAIR dialect ---------------*- C++ -*-===//
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksDialect.h"

#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksDialect.cpp.inc"
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksOps.h"
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksTypes.h"
#include "llvm/include/llvm/Support/CommandLine.h"     // from @llvm-project
#include "llvm/include/llvm/Support/InitLLVM.h"        // from @llvm-project
#include "llvm/include/llvm/Support/SourceMgr.h"       // from @llvm-project
#include "llvm/include/llvm/Support/ToolOutputFile.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"         // from @llvm-project
// #include "mlir/include/mlir/InitAllDialects.h"             // from
// @llvm-project #include "mlir/include/mlir/InitAllPasses.h"               //
// from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/include/mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Support/TypeID.h"         // from @llvm-project
// #include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from
// @llvm-project

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#define GET_TYPEDEF_CLASSES
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SCIFRCkks dialect.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace scifrckks {

void SCIFRCkksDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Backend/cornami/Dialect/SCIFRCkks/IR/SCIFRCkksOps.cpp.inc"
      >();
}
}  // namespace scifrckks
}  // namespace mlir
