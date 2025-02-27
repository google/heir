#include "lib/Target/PISA/PISAEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/PISA/IR/PISAOps.h"
#include "lib/Utils/TargetUtils.h"                       // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace pisa {

void registerToPISATranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-pisa", "translate the pisa dialect to textual pISA representation",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToPISA(op, output, false);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        tensor::TensorDialect, pisa::PISADialect,
                        mod_arith::ModArithDialect>();
      });
}

void registerToPISAInputsTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-pisa-inputs",
      "translate the pisa dialect to textual pISA representation, producing "
      "the inputs file",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToPISA(op, output, true);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        tensor::TensorDialect, pisa::PISADialect>();
      });
}

LogicalResult translateToPISA(Operation *op, llvm::raw_ostream &os,
                              bool emitInputOnly) {
  SelectVariableNames variableNames(op);
  PISAEmitter emitter(os, &variableNames, emitInputOnly);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult PISAEmitter::translate(::mlir::Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // PISA Ops
          .Case<AddOp, SubOp, MulOp, MuliOp, MacOp, MaciOp, NTTOp, INTTOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult PISAEmitter::printOperation(ModuleOp moduleOp) {
  int funcs = 0;
  for (Operation &op : moduleOp) {
    if (!llvm::isa<func::FuncOp>(op)) {
      emitError(op.getLoc(),
                "pISA emitter only supports code wrapped in functions. "
                "Operation will not be translated.");
      continue;
    }
    if (++funcs > 1)
      emitWarning(op.getLoc(),
                  "pISA emitter is designed for single functions. "
                  "Inputs, outputs and bodies of different functions "
                  "will be merged.");
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult PISAEmitter::printOperation(func::FuncOp funcOp) {
  if (emitInputOnly) {
    // TODO: Implement
    assert(false && "Not implemented yet");
    return success();
  }

  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult PISAEmitter::printOperation(func::ReturnOp op) {
  // TODO: need to map the yielded values to the outputs
  return success();
}

LogicalResult PISAEmitter::printOperation(arith::ConstantOp op) {
  // TODO: How to properly deal with constants/immediates in PISA?
  return success();
}

LogicalResult PISAEmitter::printOperation(AddOp op) {
  return printPISAOp("add", op.getResult(), {op.getLhs(), op.getRhs()},
                     op.getI());
}

LogicalResult PISAEmitter::printOperation(SubOp op) {
  return printPISAOp("sub", op.getResult(), {op.getLhs(), op.getRhs()},
                     op.getI());
}

LogicalResult PISAEmitter::printOperation(MulOp op) {
  return printPISAOp("mul", op.getResult(), {op.getLhs(), op.getRhs()},
                     op.getI());
}

LogicalResult PISAEmitter::printOperation(MuliOp op) {
  if (emitInputOnly) {
    // TODO: Implement
    return success();
  }
  auto imm = variableNames->getNameForValue(op) + "_imm";
  return printPISAOp("mul", op.getResult(), {op.getLhs()}, op.getI(), imm);
}

LogicalResult PISAEmitter::printOperation(MacOp op) {
  auto copy = printPISAOp("copy", op.getResult(), {op.getAcc()});
  if (failed(copy)) return copy;
  return printPISAOp("mac", op.getResult(), {op.getLhs(), op.getRhs()},
                     op.getI());
}

LogicalResult PISAEmitter::printOperation(MaciOp op) {
  if (emitInputOnly) {
    // TODO: Implement
    return success();
  }
  auto copy = printPISAOp("copy", op.getResult(), {op.getAcc()});
  if (failed(copy)) return copy;
  auto imm = variableNames->getNameForValue(op) + "_imm";
  return printPISAOp("mac", op.getResult(), {op.getLhs()}, op.getI(), imm);
}

LogicalResult PISAEmitter::printOperation(NTTOp op) {
  return failure();
  // TODO: how to avoid duplicating inputs for metadata?
  // TODO: How to handle double input/output from csv format?
}

LogicalResult PISAEmitter::printOperation(INTTOp op) {
  return failure();
  // TODO: how to avoid duplicating inputs for metadata?
  // TODO: How to handle double input/output from csv format?
}

LogicalResult PISAEmitter::printPISAOp(std::string_view name, Value result,
                                       ValueRange operands, int index,
                                       StringRef immediate) {
  // TODO: check if there are any duplicate occurences in operands+result
  // if there are, emit a copy operation and replace them with the copy

  os << "13, " << name << ", " << variableNames->getNameForValue(result)
     << ", ";
  os << commaSeparatedValues(operands, [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  if (!immediate.empty()) os << ", " << immediate;
  if (index >= 0) os << ", " << index;
  os << "\n";
  return success();
}

PISAEmitter::PISAEmitter(raw_ostream &os, SelectVariableNames *variableNames,
                         bool emitInputOnly)
    : os(os), emitInputOnly(emitInputOnly), variableNames(variableNames) {}

}  // namespace pisa
}  // namespace heir
}  // namespace mlir
