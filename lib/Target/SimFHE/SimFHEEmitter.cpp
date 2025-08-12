#include "lib/Target/SimFHE/SimFHEEmitter.h"

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Target/SimFHE/SimFHETemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace simfhe {

void registerToSimFHETranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-simfhe", "translate ckks dialect to SimFHE python code",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToSimFHE(op, output);
      },
      [](DialectRegistry& registry) {
        registry.insert<func::FuncDialect, ckks::CKKSDialect, lwe::LWEDialect,
                        polynomial::PolynomialDialect,
                        mod_arith::ModArithDialect, rns::RNSDialect>();
        rns::registerExternalRNSTypeInterfaces(registry);
      });
}

LogicalResult translateToSimFHE(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  SimFHEEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

SimFHEEmitter::SimFHEEmitter(llvm::raw_ostream& os,
                             SelectVariableNames* variableNames)
    : os(os), variableNames(variableNames) {}

LogicalResult SimFHEEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp, func::FuncOp, func::ReturnOp, ckks::AddOp,
                ckks::AddPlainOp, ckks::SubOp, ckks::SubPlainOp, ckks::MulOp,
                ckks::MulPlainOp, ckks::NegateOp, ckks::RotateOp,
                ckks::RelinearizeOp, ckks::RescaleOp, ckks::LevelReduceOp,
                ckks::BootstrapOp>(
              [&](auto innerOp) { return printOperation(innerOp); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });
  if (failed(status)) {
    return op.emitOpError(
        llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult SimFHEEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  SmallVector<func::FuncOp> funcs;
  for (Operation& op : moduleOp) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      funcs.push_back(func);
    }
  }
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  os << kMainPrelude << "\n";
  os.indent();  // for if __name__ == "__main__":
  os.indent();  // for internal for loop

  for (auto func : funcs) {
    os << "targets.append(Target(\"generated." << func.getName() << "\",1, [";
    for (auto arg : func.getArguments()) {
      if (isa<lwe::LWECiphertextType>(arg.getType()))
        os << "scheme_params.fresh_ctxt,";
      else if (isa<lwe::LWEPlaintextType>(arg.getType()))
        os << "params.PolyContext(scheme_params.fresh_ctxt.logq, "
              "scheme_params.fresh_ctxt.logN, "
              "scheme_params.fresh_ctxt.dnum,1), ";
      else
        return func.emitOpError("Unsupported argument type for SimFHEEmitter: ")
               << arg.getType();
    }
    os << "scheme_params]))\n";
  }
  os.unindent();  // exit internal for loop
  os << kMainEpilogue << "\n";
  return success();
}

LogicalResult SimFHEEmitter::printOperation(func::FuncOp funcOp) {
  os << "def " << funcOp.getName() << "(";
  os << heir::commaSeparatedValues(funcOp.getArguments(),
                                   [&](Value value) { return getName(value); });
  if (funcOp.getNumArguments() > 0) os << ", ";
  os << "scheme_params : params.SchemeParams):\n";
  os.indent();
  os << "stats = PerfCounter()\n";
  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }
  os << "return stats\n";
  os.unindent();
  os << "\n";
  return success();
}

LogicalResult SimFHEEmitter::printOperation(func::ReturnOp op) {
  // return is handled via accumulating stats; nothing to do
  return success();
}

#define EMIT(OPTYPE, GETTER, FUNCNAME)                       \
  LogicalResult SimFHEEmitter::printOperation(OPTYPE op) {   \
    std::string input = getName(op.GETTER());                \
    os << "stats += evaluator." FUNCNAME "(" << input        \
       << ", scheme_params.arch_param)\n";                   \
    os << getName(op.getResult()) << " = " << input << "\n"; \
    return success();                                        \
  }

#define EMIT_WITH_DROP(OPTYPE, GETTER, FUNCNAME)                    \
  LogicalResult SimFHEEmitter::printOperation(OPTYPE op) {          \
    std::string input = getName(op.GETTER());                       \
    os << "stats += evaluator." FUNCNAME "(" << input               \
       << ", scheme_params.arch_param)\n";                          \
    os << getName(op.getResult()) << " = " << input << ".drop()\n"; \
    return success();                                               \
  }

// SimFHE doesn't have `sub`. Since we only care about cost it doesn't matter.
EMIT(ckks::AddOp, getLhs, "add");
EMIT(ckks::AddPlainOp, getLhs, "add_plain");
EMIT(ckks::SubOp, getLhs, "add");
EMIT(ckks::SubPlainOp, getLhs, "add_plain");
EMIT(ckks::MulOp, getLhs, "multiply");
EMIT(ckks::MulPlainOp, getLhs, "multiply_plain");
EMIT(ckks::NegateOp, getInput, "multiply_plain");
EMIT(ckks::RotateOp, getInput, "rotate");
LogicalResult SimFHEEmitter::printOperation(ckks::RelinearizeOp op) {
  std::string name = getName(op.getInput());
  os << "stats += evaluator.key_switch(" << name << ", scheme_params.fresh_ctxt"
     << ", scheme_params.arch_param)\n";
  os << getName(op.getResult()) << " = " << name << "\n";
  op->emitWarning(
      "SimFHEEmitter currently does not know which key to use for "
      "relinearization, assuming key is similar to fresh_ctxt. ");
  return success();
}
EMIT_WITH_DROP(ckks::RescaleOp, getInput, "mod_reduce_rescale");
EMIT_WITH_DROP(ckks::LevelReduceOp, getInput, "mod_down_reduce");
EMIT(ckks::BootstrapOp, getInput, "bootstrap");

#undef UNARY_EMIT
#undef BINARY_EMIT

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir
