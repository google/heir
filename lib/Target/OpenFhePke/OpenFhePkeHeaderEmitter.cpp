#include "lib/Target/OpenFhePke/OpenFhePkeHeaderEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
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

LogicalResult translateToOpenFhePkeHeader(Operation* op, llvm::raw_ostream& os,
                                          OpenfheImportType importType) {
  SelectVariableNames variableNames(op);
  OpenFhePkeHeaderEmitter emitter(os, &variableNames, importType);
  return emitter.translate(*op);
}

LogicalResult OpenFhePkeHeaderEmitter::translate(Operation& op) {
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

LogicalResult OpenFhePkeHeaderEmitter::printOperation(ModuleOp moduleOp) {
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

  os << getModulePrelude(scheme, importType_) << "\n";
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult OpenFhePkeHeaderEmitter::printOperation(func::FuncOp funcOp) {
  auto res = funcDeclarationHelper(
      funcOp, os, variableNames,
      [&](Type type, Location loc) { return emitType(type, loc); },
      [&](Location loc, const std::string& message) {
        return emitError(loc, message);
      });
  if (failed(res)) {
    return res;
  }
  os << ";\n";
  os.unindent();
  return success();
}

LogicalResult OpenFhePkeHeaderEmitter::emitType(Type type, Location loc) {
  auto result = convertType(type, loc);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

OpenFhePkeHeaderEmitter::OpenFhePkeHeaderEmitter(
    raw_ostream& os, SelectVariableNames* variableNames,
    OpenfheImportType importType)
    : importType_(importType), os(os), variableNames(variableNames) {}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
