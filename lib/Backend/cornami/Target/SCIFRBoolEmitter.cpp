#include "lib/Backend/cornami/Target/SCIFRBoolEmitter.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "SCIFRBoolEmitter.h"
#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"
#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"
#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolTypes.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

using namespace mlir::scifrbool;

namespace mlir {
namespace cornami {
namespace target {

struct TranslateOptions {
  llvm::cl::opt<std::string> packageName{
      "emit-scifrbool", llvm::cl::desc("translate the SCIFRBool dialect to C++ "
                                       "code against the Concrete Engine "
                                       "API")};
};

static llvm::ManagedStatic<TranslateOptions> translateOptions;

void registerTranslateOptions() {
  // Forces initialization of options.
  *translateOptions;
}

void registerToSCIFRBoolTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-scifrbool",
      "translate the SCIFRBool dialect to C++ code against the Concrete Engine "
      "API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToSCIFRBool(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<scifrbool::SCIFRBoolDialect, arith::ArithDialect,
                        func::FuncDialect, tensor::TensorDialect,
                        heir::lwe::LWEDialect, heir::cggi::CGGIDialect,
                        memref::MemRefDialect,
                        mlir::heir::mod_arith::ModArithDialect>();
      });
}

LogicalResult translateToSCIFRBool(Operation *op, llvm::raw_ostream &os) {
  SCIFRBoolEmitter emitter(os);
  emitter.autoGenComment();
  emitter.beginCode();
  LogicalResult result = emitter.translate(*op);
  return result;
}

// TODO: check if more operators need to be added
LogicalResult SCIFRBoolEmitter::translate(Operation &op) {
  createVariableNames(&op);
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return success(); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return success(); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // SCIFRBool ops
          .Case<AndOp, NandOp, OrOp, NorOp, NotOp, XorOp, XNorOp, KSOp,
                LinearOp, PBSOp, SectionOp>(
              [&](auto op) { return printOperation(op); })
          .Case<tensor::FromElementsOp, tensor::ExtractOp>(
              [&](auto op) { return printOperation(op); })
          // MemRef ops
          .Case<memref::AllocOp, memref::LoadOp, memref::StoreOp>(
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

LogicalResult SCIFRBoolEmitter::printOperation(ModuleOp moduleOp) {
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(func::FuncOp funcOp) {
  if (failed(canEmitFuncForSCIFRBool(funcOp))) {
    // Return success implies print nothing, and note the called function
    // emits a warning.
    return success();
  }
  if (funcOp.getNumResults() != 1) {
    return funcOp.emitOpError() << "Only functions with a single return type "
                                   "are supported, but this function has "
                                << funcOp.getNumResults();
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result))) {
    return funcOp.emitOpError() << "Failed to emit type " << result;
  }

  m_os << " " << funcOp.getName() << "(";
  m_os.indent();

  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType()))) {
      return funcOp.emitOpError() << "Failed to emit type " << arg.getType();
    }
  }

  m_os << heir::commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    return convertType(value.getType()).value() + " " + getNameForValue(value);
  });
  m_os.unindent();
  m_os << ") {\n";
  m_os.indent();

  for (Value arg : funcOp.getArguments()) {
    std::string argType = convertType(arg.getType()).value();
    if (argType == "BootstrapKeyStandard<uint64_t>") {
      m_os << argType << " bsk = " << getNameForValue(arg) << ";\n";
    } else if (argType == "KeySwitchKey<uint64_t>") {
      m_os << argType << " ksk = " << getNameForValue(arg) << ";\n";
    } else if (argType == "ServerParameters") {
      m_os << argType << " serverSetup = " << getNameForValue(arg) << ";\n";
    }
  }
  // create concrete engine
  m_os << "ConcreteEngine<uint64_t>* pEngine = new "
          "ConcreteEngine<uint64_t>(bsk, ksk, serverSetup);\n"
       << "ConcreteEngineBoolean<uint64_t> engineBoolean(pEngine, "
          "BooleanEncodingType::NegativeOneOne);\n"
       << "pEngine = NULL;\n\n";

  m_argList.clear();
  for (Value arg : funcOp.getArguments()) {
    std::string argType = convertType(arg.getType()).value();
    if (m_scifrTypes.count(argType)) continue;
    if (llvm::isa<RankedTensorType>(arg.getType())) continue;
    m_os << "StreamLwe<uint64_t> strm" << m_strmVarCount
         << " = engineBoolean.newHostToFabricLweStream();\n";
    m_oprToStrm[getNameForValue(arg)].push_back("strm" +
                                                std::to_string(m_strmVarCount));
    m_strmVarCount++;
    m_argList.push_back(arg);
  }
  m_os << "uint nLweSize = bsk.m_bskInfo.lweInfo.nLWESize;\n";

  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }
  m_os << m_returnStr;
  m_fromElementsStr.clear();
  m_returnStr.clear();

  m_os.unindent();
  m_os << "}\n";
  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() != 1) {
    op.emitError() << "Only one return value supported";
    return failure();
  }
  m_returnStr = "";
  std::string returnVar = getNameForValue(op.getOperands()[0]);
  m_returnStr += m_fromElementsStr;
  m_returnStr += "engineBoolean.m_engine->profiler_report();\n";
  m_returnStr += "return " + returnVar + ";\n";
  return success();
}

LogicalResult SCIFRBoolEmitter::printSksMethod(::mlir::ValueRange operands,
                                               ::mlir::Value result,
                                               std::string_view operationName) {
  std::string argList = "";
  for (Value opr : operands) {
    std::string opName = getNameForValue(opr);
    createStream(opName);
    m_isConsumed.insert(m_oprToStrm[opName].back());
    if (argList != "") argList += ", ";
    argList += m_oprToStrm[opName].back();
  }
  std::string opName = getNameForValue(result);
  createStream(opName);
  if (argList != "") argList += ", ";
  argList += m_oprToStrm[opName].back();

  m_os << "engineBoolean.binaryBooleanConsumeAll(" << argList
       << ", BinaryBooleanOp::" << operationName << ", 1);\n";
  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(NotOp op) {
  std::string argList = "";
  std::string inputName = getNameForValue(op.getInput());
  createStream(inputName);
  m_isConsumed.insert(m_oprToStrm[inputName].back());
  if (argList != "") argList += ", ";
  argList += m_oprToStrm[inputName].back();
  std::string resultName = getNameForValue(op.getResult());
  createStream(resultName);
  if (argList != "") argList += ", ";
  argList += m_oprToStrm[resultName].back();

  m_os << "engineBoolean.unaryBooleanConsumeAll(" << argList
       << ", UnaryBooleanOp::Not, 1);\n";
  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(AndOp op) {
  return printSksMethod(op.getOperands(), op.getResult(), "And");
}

LogicalResult SCIFRBoolEmitter::printOperation(NandOp op) {
  return printSksMethod(op.getOperands(), op.getResult(), "Nand");
}

LogicalResult SCIFRBoolEmitter::printOperation(OrOp op) {
  return printSksMethod(op.getOperands(), op.getResult(), "Or");
}

LogicalResult SCIFRBoolEmitter::printOperation(NorOp op) {
  return printSksMethod(op.getOperands(), op.getResult(), "Nor");
}

LogicalResult SCIFRBoolEmitter::printOperation(XorOp op) {
  return printSksMethod(op.getOperands(), op.getResult(), "Xor");
}

LogicalResult SCIFRBoolEmitter::printOperation(XNorOp op) {
  return printSksMethod(op.getOperands(), op.getResult(), "Xnor");
}

LogicalResult SCIFRBoolEmitter::printOperation(KSOp op) {
  return printSksMethod({op.getInput()}, op.getResult(), "KS");
}

LogicalResult SCIFRBoolEmitter::printOperation(LinearOp op) {
  return printSksMethod({op.getInput()}, op.getResult(), "Linear");
}

LogicalResult SCIFRBoolEmitter::printOperation(PBSOp op) {
  return printSksMethod({op.getInput()}, op.getResult(), "PBS");
}

LogicalResult SCIFRBoolEmitter::printOperation(SectionOp op) {
  m_os << "// =================== start of section " << m_sectionCount + 1
       << " ===================\n\n";
  if (m_sectionCount > 0)
    m_os << "engineBoolean.m_engine->resetTopology(true);\n";
  m_os << "engineBoolean.m_engine->SetDefaultDirTopology(\"/tmp/viz/"
       << m_sectionCount << "\");\n";

  std::string sectionOpResultName = getNameForValue(op.getResult(0));
  op.walk([&](Operation *opr) {
    llvm::TypeSwitch<Operation &>(*opr)
        .Case<scifrbool::AndOp, scifrbool::OrOp, scifrbool::NotOp,
              scifrbool::NandOp, scifrbool::NorOp, scifrbool::XorOp,
              scifrbool::XNorOp>([&](auto operation) {
          if (failed(printOperation(operation))) {
            op.emitError() << "Creating Section Op failed";
          }
          std::string resultName = getNameForValue(operation.getResult());
          m_oprToStrm[sectionOpResultName].push_back(
              m_oprToStrm[resultName].back());
        })
        .Default([&](Operation &) {
          // Do nothing
        });
  });
  m_os << "engineBoolean.m_engine->routeToHost("
       << m_oprToStrm[sectionOpResultName].back() << ");\n";
  m_os << "engineBoolean.m_engine->finalizeAndRun();\n";
  for (Value arg : op.getOperands()) {
    std::string argType = convertType(arg.getType()).value();
    std::string argName = getNameForValue(arg);
    if (llvm::isa<RankedTensorType>(arg.getType())) {
      for (auto extractedVal : m_extractedValues[argName]) {
        m_os << "engineBoolean.m_engine->put("
             << m_oprToStrm[extractedVal].back() << ", &" << extractedVal
             << "[0], nLweSize);\n";
      }
    } else {
      m_os << "engineBoolean.m_engine->put(" << m_oprToStrm[argName].back()
           << ", &" << argName << "[0], nLweSize);\n";
    }
  }
  m_os << convertType(op.getOperands()[0].getType()).value() << " "
       << sectionOpResultName << " = std::vector<uint64_t>(nLweSize, 0);\n";
  m_os << "engineBoolean.m_engine->get("
       << m_oprToStrm[sectionOpResultName].back() << ", &"
       << sectionOpResultName << "[0], nLweSize);\n";
  m_os << "engineBoolean.m_engine->EndRun();\n";
  m_os << "// =================== end of section " << m_sectionCount + 1
       << " ===================\n\n";
  m_sectionCount++;
  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(tensor::FromElementsOp op) {
  m_fromElementsStr = "";
  std::string resultVar = getNameForValue(op.getResult());
  m_fromElementsStr +=
      convertType(op.getResult().getType()).value() + " " + resultVar + " = ";
  m_fromElementsStr +=
      "{" +
      heir::commaSeparatedValues(
          op.getOperands(),
          [&](Value value) { return getNameForValue(value); }) +
      "};\n";
  return success();
}

// Produces a SCIFRBoolCiphertext
LogicalResult SCIFRBoolEmitter::printOperation(tensor::ExtractOp op) {
  // Assuming the indices to be SSA values (not integer attributes)
  if (failed(emitTypedAssignPrefix(op.getResult()))) {
    return failure();
  }
  std::string tensorName = getNameForValue(op.getTensor()),
              resultName = getNameForValue(op.getResult());
  m_os << tensorName << "["
       << heir::commaSeparatedValues(
              op.getIndices(),
              [&](Value value) { return getNameForValue(value); })
       << "];\n";
  m_os << "StreamLwe<uint64_t> strm" << m_strmVarCount
       << " = engineBoolean.newHostToFabricLweStream();\n";
  m_oprToStrm[resultName].push_back("strm" + std::to_string(m_strmVarCount));
  m_strmVarCount++;
  m_extractedValues[tensorName].push_back(resultName);
  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(memref::AllocOp op) {
  m_os << "std::vector<";
  if (failed(emitType(op.getMemref().getType().getElementType()))) {
    return op.emitOpError() << "Failed to get memref element type";
  }
  std::string resultVar = getNameForValue(op.getMemref());
  m_os << "> " << resultVar << ";\n";

  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(memref::LoadOp op) {
  if (isa<BlockArgument>(op.getMemref())) {
    if (failed(emitTypedAssignPrefix(op.getResult()))) {
      return failure();
    }
    m_os << getNameForValue(op.getMemRef()) << "["
         << getNameForValue(op.getIndices()[0]) << "];\n";
  }
  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(memref::StoreOp op) {
  m_os << getNameForValue(op.getMemref());
  m_os << ".push_back(" << getNameForValue(op.getValueToStore()) << ");\n";

  return success();
}

LogicalResult SCIFRBoolEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    if (failed(emitTypedAssignPrefix(op.getResult()))) {
      return failure();
    }
    m_os << intAttr.getValue() << ";\n";
  } else {
    return op.emitError() << "Unknown constant type " << valueAttr.getType();
  }
  return success();
}

void SCIFRBoolEmitter::createStream(std::string opName) {
  if (m_oprToStrm.find(opName) != m_oprToStrm.end()) {
    // TODO: Check how to create new stream from Consumed input streams
    if (m_isConsumed.find(m_oprToStrm[opName].back()) !=
        m_isConsumed.end()) {  // if the stream has been consumed, clone it
      // m_os << "StreamLwe<uint64_t> strm" << m_strmVarCount
      //      << " = engineBoolean.cloneStream(" << m_oprToStrm[opName][0] <<
      //      ");\n";
      m_os << "StreamLwe<uint64_t> strm" << m_strmVarCount
           << " = engineBoolean.newLweStream();\n";
      m_oprToStrm[opName].push_back("strm" + std::to_string(m_strmVarCount));
    } else {  // No need to create new stream
      return;
    }
  } else {
    m_os << "StreamLwe<uint64_t> strm" << m_strmVarCount
         << " = engineBoolean.newLweStream();\n";
    m_oprToStrm[opName].push_back("strm" + std::to_string(m_strmVarCount));
  }
  m_strmVarCount++;
}

void SCIFRBoolEmitter::autoGenComment() {
  m_os << R"cpp(/*=================================================================
//
//  This file has been Automatically generated by SCIFRBool C++ Emitter
//
=================================================================*/
  )cpp";
}

void SCIFRBoolEmitter::errorCheck() {
  m_os << "if (nResult != 0) {\n";
  m_os.indent();
  m_os << "return -1;\n";
  m_os.unindent();
  m_os << "}\n";
}

void SCIFRBoolEmitter::beginCode() {
  m_os << R"cpp(#include <concrete_engine.hpp>
#include <concrete_engine_boolean.hpp>

                using cornami::fhe::ServerParameters;
                using cornami::fhe::concrete::BootstrapKeyStandard;
                using cornami::fhe::concrete::ConcreteEngine;
                using cornami::fhe::concrete::KeySwitchKey;
                using cornami::fhe::concrete::StreamLwe;

                using cornami::fhe::concrete::BinaryBooleanOp;
                using cornami::fhe::concrete::BooleanEncodingType;
                using cornami::fhe::concrete::ConcreteEngineBoolean;
                using cornami::fhe::concrete::UnaryBooleanOp;

                using SCIFRBoolCiphertext = std::vector<uint64_t>;
  )cpp";
}

FailureOr<std::string> SCIFRBoolEmitter::convertType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    // A lambda in a type switch statement can't return multiple types.
    // FIXME: why can't both types be FailureOr<std::string>?
    auto elementTy = convertType(shapedType.getElementType());
    if (failed(elementTy)) return failure();

    return std::string(std::string("std::vector<") + elementTy.value() + ">");
  }
  return llvm::TypeSwitch<Type &, FailureOr<std::string>>(type)
      .Case<SCIFRBoolCiphertextType>(
          [&](auto ty) { return std::string("SCIFRBoolCiphertext"); })
      .Case<IndexType>([&](auto ty) { return std::string("size_t"); })
      .Case<IntegerType>([&](auto ty) {
        auto width = ty.getWidth();
        if (width != 8 && width != 16 && width != 32 && width != 64) {
          return FailureOr<std::string>();
        }
        SmallString<8> result;
        llvm::raw_svector_ostream os(result);
        os << "int" << width << "_t";
        return FailureOr<std::string>(std::string(result));
      })
      .Case<RankedTensorType>([&](auto ty) {
        if (ty.getRank() != 1) {
          return FailureOr<std::string>();
        }

        auto eltTyResult = convertType(ty.getElementType());
        if (failed(eltTyResult)) {
          return FailureOr<std::string>();
        }

        SmallString<8> result;
        llvm::raw_svector_ostream tmp_os(result);
        tmp_os << "std::vector<" << eltTyResult.value() << ">";
        return FailureOr<std::string>(std::string(result));
      })
      .Case<SCIFRBoolBootstrapKeyStandardType>([&](auto ty) {
        return std::string("BootstrapKeyStandard<uint64_t>");
      })
      .Case<SCIFRBoolKeySwitchKeyType>(
          [&](auto ty) { return std::string("KeySwitchKey<uint64_t>"); })
      .Case<SCIFRBoolServerParametersType>(
          [&](auto ty) { return std::string("ServerParameters"); })
      .Default([&](Type &) { return failure(); });
}

LogicalResult SCIFRBoolEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  m_os << result;
  return success();
}

LogicalResult SCIFRBoolEmitter::emitTypedAssignPrefix(Value result) {
  if (failed(emitType(result.getType()))) {
    return failure();
  }
  m_os << " " << getNameForValue(result) << " = ";
  return success();
}

SCIFRBoolEmitter::SCIFRBoolEmitter(raw_ostream &os) : m_os(os) {}

}  // namespace target
}  // namespace cornami
}  // namespace mlir
