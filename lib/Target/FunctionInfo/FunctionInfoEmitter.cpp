#include "lib/Target/FunctionInfo/FunctionInfoEmitter.h"

#include <cstdint>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "llvm/include/llvm/ADT/StringRef.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace functioninfo {

void registerToFunctionInfoTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-function-info", "Emit function info (helper for python frontend)",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToFunctionInfo(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, secret::SecretDialect>();
      });
}

LogicalResult translateToFunctionInfo(Operation *op, llvm::raw_ostream &os) {
  FunctionInfoEmitter emitter(os);
  return emitter.translate(*op);
}

FunctionInfoEmitter::FunctionInfoEmitter(llvm::raw_ostream &os) : os(os) {}

LogicalResult FunctionInfoEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          .Case<ModuleOp>([&](auto innerOp) { return printOperation(innerOp); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });
  if (failed(status)) {
    return op.emitOpError(
        llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult FunctionInfoEmitter::printOperation(ModuleOp moduleOp) {
  // Secretness Annotation:
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();

  auto result = solver.initializeAndRun(moduleOp);

  if (failed(result)) {
    moduleOp.emitOpError() << "Failed to run the analysis.\n";
    return failure();
  }

  auto funcs = moduleOp.getOps<func::FuncOp>();
  if (funcs.empty()) {
    moduleOp.emitOpError("No functions found in the module");
    return failure();
  }

  // for now, we only print the first function
  auto funcOp = *funcs.begin();

  // We probably needed to --mlir-print-generic the input anyway,
  // as this emitter doesn't have all the dialects registered
  // we can just print arg_0, arg_1, etc.
  // TODO (#1162): Investigate how we can preserve custom ssa names
  os << funcOp.getSymName() << "\n";
  for (int i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    os << "arg_" << i;
    if (i < e - 1) os << ", ";
  }

  os << "\n";

  // Now emit the indices of the secret arguments:
  SmallVector<int64_t> secretIndices;
  for (auto arg : funcOp.getArguments())
    if (isSecret(arg, &solver)) secretIndices.push_back(arg.getArgNumber());

  for (int64_t i = 0, e = secretIndices.size(); i < e; ++i) {
    os << secretIndices[i];
    if (i < e - 1) os << ", ";
  }

  os << "\n";

  return success();
}

}  // namespace functioninfo
}  // namespace heir
}  // namespace mlir
