#include "lib/Target/Poulpy/PoulpyEmitter.h"

#include "lib/Dialect/Poulpy/IR/PoulpyDialect.h"
#include "lib/Dialect/Poulpy/IR/PoulpyOps.h"
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.h"
#include "lib/Target/Poulpy/PoulpyTemplates.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
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

std::string valueOrClonedValue(Value value,
                               SelectVariableNames* variableNames) {
  auto expression = variableNames->getNameForValue(value);
  if (isa<BlockArgument>(value)) {
    expression += ".clone()";
  }
  return expression;
}

std::string ref(Value value, SelectVariableNames* variableNames) {
  auto expression = variableNames->getNameForValue(value);
  return isa<BlockArgument>(value) ? "&*" + expression : "&" + expression;
}

std::string refMut(Value value, SelectVariableNames* variableNames) {
  auto expression = variableNames->getNameForValue(value);
  return isa<BlockArgument>(value) ? "&mut *" + expression
                                   : "&mut " + expression;
}
}  // namespace

void registerToPoulpyTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-poulpy", "translate the poulpy dialect to Rust code for poulpy",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToPoulpy(op, output);
      },
      [](DialectRegistry& registry) {
        registry.insert<func::FuncDialect, memref::MemRefDialect,
                        poulpy::PoulpyDialect>();
      });
}

LogicalResult translateToPoulpy(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  PoulpyEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult PoulpyEmitter::translateBlock(Block& block) {
  for (Operation& op : block.getOperations()) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult PoulpyEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          // TODO(mmoro): have all of them in the same .Case?
          // Builtin ops
          .Case<ModuleOp>([&](ModuleOp op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          .Case<AddOp, AddAssignOp>([&](auto op) { return printOperation(op); })
          .Case<SubOp, SubAssignOp>([&](auto op) { return printOperation(op); })
          .Case<MulOp, MulAssignOp>([&](auto op) { return printOperation(op); })
          .Case<memref::AllocOp>(
              [&](memref::AllocOp op) { return printOperation(op); })
          .Default([&](Operation& op) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }

  return success();
}

void PoulpyEmitter::computeMutatedValues(func::FuncOp funcOp) {
  mutatedValues.clear();

  funcOp.walk([&](Operation* op) {
    llvm::TypeSwitch<Operation&, void>(*op)
        .Case<AddOp, AddAssignOp, SubOp, SubAssignOp, MulOp, MulAssignOp>(
            [&](auto op) { mutatedValues.insert(op.getDst()); })
        .Default([&](Operation& op) {});
  });
}

void PoulpyEmitter::materializeIfPending(Value dst, Value module,
                                         Value layoutSource) {
  if (pendingAllocs.erase(dst)) {
    auto layoutName = variableNames->getNameForValue(layoutSource);
    auto dstName = variableNames->getNameForValue(dst);
    auto moduleName = variableNames->getNameForValue(module);
    os << "let mut " << dstName << " = " << moduleName
       << ".ckks_ciphertext_alloc(" << layoutName << ".base2k(), " << layoutName
       << ".max_k());\n";
  }
}

LogicalResult PoulpyEmitter::checkNotPending(Value dst, Operation* op) {
  if (pendingAllocs.contains(dst)) {
    InFlightDiagnostic diag =
        op->emitError("dst has not been initialized before this use");
    diag.attachNote(dst.getLoc())
        << "allocated here but never written to first";
    return diag;
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
  os << kTypeAliases << "\n";
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult PoulpyEmitter::printOperation(func::FuncOp funcOp) {
  computeMutatedValues(funcOp);
  os << "pub fn " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    // TODO(mmoro): add type, should we check integertype like tfherust? how to
    // handle reference?
    os << argName << ": ";
    bool isMutated = mutatedValues.contains(arg);
    if (failed(emitType(arg.getType(), true, isMutated))) {
      return funcOp.emitOpError()
             << "Failed to emit poulpy type " << arg.getType();
    }
    os << ",\n";
  }
  os.unindent();
  os << ") -> Result<";

  auto numResults = funcOp.getNumResults();
  if (numResults == 0) {
    os << "()";
  } else if (numResults == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result, false, false))) {
      return funcOp.emitOpError() << "Failed to emit poulpy type " << result;
    }
  } else {
    // TODO(mmoro): implement
  }

  os << "> {\n";
  os.indent();
  for (Block& block : funcOp.getBlocks()) {
    if (failed(translateBlock(block))) {
      return funcOp.emitOpError()
             << "Failed to translate block of func " << funcOp.getName();
    }
  }

  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult PoulpyEmitter::printOperation(func::ReturnOp op) {
  // TODO(mmoro): implement ReturnOp printing for non-zero number of return
  // values
  if (op.getNumOperands() == 0) {
    os << "Ok(())\n";
  } else if (op.getNumOperands() == 1) {
    auto returnOperand = op.getOperands()[0];
    auto expression = valueOrClonedValue(returnOperand, variableNames);
    os << "Ok(" << expression << ")\n";
  }
  return success();
}

LogicalResult PoulpyEmitter::printOperation(AddOp addOp) {
  auto module = addOp.getModule();
  auto dst = addOp.getDst();
  auto a = addOp.getA();
  auto b = addOp.getB();
  auto scratch = addOp.getScratch();

  materializeIfPending(dst, module, a);

  os << variableNames->getNameForValue(module) << ".ckks_add_into("
     << refMut(dst, variableNames) << ", " << ref(a, variableNames) << ", "
     << ref(b, variableNames) << ", " << "&mut "
     << variableNames->getNameForValue(scratch) << ".borrow())?;\n";

  return success();
}
LogicalResult PoulpyEmitter::printOperation(AddAssignOp addAssignOp) {
  auto module = addAssignOp.getModule();
  auto dst = addAssignOp.getDst();
  auto a = addAssignOp.getA();
  auto scratch = addAssignOp.getScratch();

  if (failed(checkNotPending(dst, addAssignOp))) return failure();

  os << variableNames->getNameForValue(module) << ".ckks_add_assign("
     << refMut(dst, variableNames) << ", " << ref(a, variableNames) << ", "
     << "&mut " << variableNames->getNameForValue(scratch) << ".borrow())?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(SubOp subOp) {
  auto module = subOp.getModule();
  auto dst = subOp.getDst();
  auto a = subOp.getA();
  auto b = subOp.getB();
  auto scratch = subOp.getScratch();

  materializeIfPending(dst, module, a);

  os << variableNames->getNameForValue(module) << ".ckks_sub_into("
     << refMut(dst, variableNames) << ", " << ref(a, variableNames) << ", "
     << ref(b, variableNames) << ", " << "&mut "
     << variableNames->getNameForValue(scratch) << ".borrow())?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(SubAssignOp subAssignOp) {
  auto module = subAssignOp.getModule();
  auto dst = subAssignOp.getDst();
  auto a = subAssignOp.getA();
  auto scratch = subAssignOp.getScratch();

  if (failed(checkNotPending(dst, subAssignOp))) return failure();

  os << variableNames->getNameForValue(module) << ".ckks_sub_assign("
     << refMut(dst, variableNames) << ", " << ref(a, variableNames) << ", "
     << "&mut " << variableNames->getNameForValue(scratch) << ".borrow())?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(MulOp mulOp) {
  auto module = mulOp.getModule();
  auto dst = mulOp.getDst();
  auto a = mulOp.getA();
  auto b = mulOp.getB();
  auto tsk = mulOp.getTsk();
  auto scratch = mulOp.getScratch();

  materializeIfPending(dst, module, a);

  os << variableNames->getNameForValue(module) << ".ckks_mul_into("
     << refMut(dst, variableNames) << ", " << ref(a, variableNames) << ", "
     << ref(b, variableNames) << ", " << ref(tsk, variableNames) << ", "
     << "&mut " << variableNames->getNameForValue(scratch) << ".borrow())?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(MulAssignOp mulAssignOp) {
  auto module = mulAssignOp.getModule();
  auto dst = mulAssignOp.getDst();
  auto a = mulAssignOp.getA();
  auto tsk = mulAssignOp.getTsk();
  auto scratch = mulAssignOp.getScratch();

  if (failed(checkNotPending(dst, mulAssignOp))) return failure();

  os << variableNames->getNameForValue(module) << ".ckks_mul_assign("
     << refMut(dst, variableNames) << ", " << ref(a, variableNames) << ", "
     << ref(tsk, variableNames) << ", "
     << "&mut " << variableNames->getNameForValue(scratch) << ".borrow())?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(memref::AllocOp allocOp) {
  pendingAllocs.insert(allocOp.getResult());
  return success();
}

FailureOr<std::string> PoulpyEmitter::convertType(Type type, bool isArg,
                                                  bool isMutated) {
  return llvm::TypeSwitch<Type&, FailureOr<std::string>>(type)
      .Case<ModuleType>([&](ModuleType) -> FailureOr<std::string> {
        return std::string("&Module<BE>");
      })
      .Case<ScratchType>([&](ScratchType) -> FailureOr<std::string> {
        return std::string("&mut ScratchOwned<BE>");
      })
      .Case<MemRefType>([&](MemRefType memRefType) -> FailureOr<std::string> {
        if (memRefType.getRank() != 0) return failure();
        CiphertextType ciphertextType =
            dyn_cast<CiphertextType>(memRefType.getElementType());
        if (!ciphertextType) return failure();
        return std::string(isArg ? (isMutated ? "&mut Ct" : "&Ct") : "Ct");
      })
      .Case<TensorKeyType>([&](TensorKeyType) -> FailureOr<std::string> {
        return std::string("&Tsk");
      })
      .Default([&](Type&) { return failure(); });
}

LogicalResult PoulpyEmitter::emitType(Type type, bool isArg, bool isMutated) {
  auto result = convertType(type, isArg, isMutated);
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
