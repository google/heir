#ifndef SCIFRCkks_SCIFRCkksDialect_H
#define SCIFRCkks_SCIFRCkksDialect_H

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

// clang-format off
#include "lib/Dialect/SCIFRCkks/IR/SCIFRCkksDialect.h.inc"
// clang-format on

#endif  // SCIFRCkks_SCIFRCkksDialect_H
