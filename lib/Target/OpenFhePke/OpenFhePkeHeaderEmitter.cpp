#include "lib/Target/OpenFhePke/OpenFhePkeHeaderEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/ModuleAttributes.h"
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
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
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
  // If keeping this consistent alongside OpenFheEmitter gets annoying,
  // extract to a shared function in a base class.
  if (funcOp.getNumResults() != 1) {
    return funcOp.emitOpError() << "Only functions with a single return type "
                                   "are supported, but this function has "
                                << funcOp.getNumResults();
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result, funcOp->getLoc()))) {
    return funcOp.emitOpError() << "Failed to emit type " << result;
  }

  os << " " << funcOp.getName() << "(";

  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType(), arg.getLoc()))) {
      return funcOp.emitOpError() << "Failed to emit type " << arg.getType();
    }
  }

  os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    auto res = convertType(value.getType(), funcOp->getLoc());
    return res.value() + " " + variableNames->getNameForValue(value);
  });
  os << ");\n";

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
