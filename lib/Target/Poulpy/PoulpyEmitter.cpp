#include "lib/Target/Poulpy/PoulpyEmitter.h"

#include "lib/Dialect/Poulpy/IR/PoulpyDialect.h"
#include "lib/Dialect/Poulpy/IR/PoulpyOps.h"
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.h"
#include "lib/Target/Poulpy/PoulpyTemplates.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

#define DEBUG_TYPE "poulpy-emitter"

namespace mlir {
namespace heir {
namespace poulpy {

namespace {
FailureOr<std::string> detectBackend(ModuleOp* op) {
  std::optional<PoulpyBackend> found;
  for (auto funcOp : op->getOps<func::FuncOp>()) {
    // TODO(mmoro): also check result types for backend information
    for (Type argType : funcOp.getArgumentTypes()) {
      auto moduleType = dyn_cast<ModuleType>(argType);
      if (!moduleType) continue;
      PoulpyBackend backend = moduleType.getBackend();
      if (found.has_value() && found != backend) {
        return op->emitError("poulpy module contains multiple backends");
      }
      found = backend;
    }
  }
  switch (found.value_or(PoulpyBackend::FFT64Ref)) {
    case PoulpyBackend::FFT64Ref:
      return std::string("FFT64Ref");
    case PoulpyBackend::NTT4x30Ref:
      return std::string("NTT4x30Ref");
  }
  llvm_unreachable("unhandled PoulpyBackend");
}
}  // namespace

void registerToPoulpyTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-poulpy", "translate the poulpy dialect to Rust code for poulpy",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToPoulpy(op, output);
      },
      [](DialectRegistry& registry) {
        registry.insert<func::FuncDialect, poulpy::PoulpyDialect>();
      });
}

LogicalResult translateToPoulpy(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  PoulpyEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult PoulpyEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp>([&](ModuleOp op) { return printOperation(op); })
          .Default([&](Operation& op) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }

  return success();
}

LogicalResult PoulpyEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  auto backend = detectBackend(&moduleOp);
  if (failed(backend)) {
    moduleOp.emitOpError("Error while detecting backend");
    return failure();
  }
  os << "type BE = " << backend.value() << ";\n";
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

FailureOr<std::string> PoulpyEmitter::convertType(Type type) {
  // TODO(mmoro): implement type conversion
  return std::string("TODO");
}

LogicalResult PoulpyEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}
PoulpyEmitter::PoulpyEmitter(raw_ostream& os,
                             SelectVariableNames* variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace poulpy
}  // namespace heir
}  // namespace mlir