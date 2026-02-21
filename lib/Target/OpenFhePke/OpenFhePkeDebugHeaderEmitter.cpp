#include "lib/Target/OpenFhePke/OpenFhePkeDebugHeaderEmitter.h"

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

LogicalResult translateToOpenFhePkeDebugHeaderEmitter(
    Operation* op, llvm::raw_ostream& os, OpenfheImportType importType) {
  OpenFhePkeDebugHeaderEmitter emitter(os, importType);
  return emitter.translate(*op);
}

LogicalResult OpenFhePkeDebugHeaderEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFhePkeDebugHeaderEmitter::printOperation(ModuleOp moduleOp) {
  OpenfheScheme scheme;
  if (moduleIsBGV(moduleOp)) {
    scheme = OpenfheScheme::BGV;
  } else if (moduleIsBFV(moduleOp)) {
    scheme = OpenfheScheme::BFV;
  } else if (moduleIsCKKS(moduleOp)) {
    scheme = OpenfheScheme::CKKS;
  } else {
    return emitError(moduleOp.getLoc(), "Missing scheme attribute on module");
  }

  os << KdebugHeaderImports << "\n";
  os << getModulePrelude(scheme, importType_) << "\n";

  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult OpenFhePkeDebugHeaderEmitter::printOperation(
    func::FuncOp funcOp) {
  if ((!isDebugPort(funcOp.getName())) || (isEmitted)) {
    return success();
  }

  auto res = emitDebugHelperSignature(
      funcOp, os, [&](Location loc, const std::string& message) {
        return emitError(loc, message);
      });

  if (failed(res)) {
    return res;
  }

  os << ";\n";
  os.unindent();
  isEmitted = true;
  return success();
}

OpenFhePkeDebugHeaderEmitter::OpenFhePkeDebugHeaderEmitter(
    raw_ostream& os, OpenfheImportType importType)
    : importType_(importType), os(os), isEmitted(false) {}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
