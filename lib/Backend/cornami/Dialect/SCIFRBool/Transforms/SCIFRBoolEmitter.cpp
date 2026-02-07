#include "lib/Backend/cornami/Dialect/SCIFRBool/Transforms/SCIFRBoolEmitter.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"
#include "lib/Backend/cornami/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

// clang-format off
#include "lib/Backend/cornami/Dialect/SCIFRBool/Transforms/SCIFRBoolEmitter.h.inc"
// clang-format on

namespace mlir {
namespace cornami {

#define GEN_PASS_DEF_CGGIEMITTER
#include "lib/Backend/cornami/Dialect/SCIFRBool/Transforms/SCIFRBoolEmitter.h.inc"

void populateDims(mlir::TensorType mlirTensorType, EMITTER_NODE& node) {
  std::vector<int> vDims;
  for (auto iCtr = mlirTensorType.getShape().begin();
       iCtr != mlirTensorType.getShape().end(); iCtr++) {
    vDims.push_back(*iCtr);
  }
  node.dims = vDims;
}

struct CGGIEmitter : impl::CGGIEmitterBase<CGGIEmitter> {
  using CGGIEmitterBase::CGGIEmitterBase;

  void runOnOperation() override {
    SCIFRBoolEmitter emit;
    emit.m_codestr.open("scifrbool_output_code.cpp");
    emit.BeginCode();

    std::unordered_map<std::string, std::string> oprToStrm;
    int strmVarCount = 0;
    auto result = getOperation()->walk([&](Operation* op) {
      std::string argList = "";
      std::vector<std::vector<std::string>> opCommands;
      std::vector<std::string> opOprnds;
      for (int itr = 0; itr < (int)op->getOperands().size(); itr++) {
        if (op->getOperand(itr).getType().isIntOrIndexOrFloat()) {
          arith::ConstantOp constOp =
              dyn_cast<arith::ConstantOp>(op->getOperand(itr).getDefiningOp());
          continue;
        }
        EMITTER_NODE node = emit.ProcessOperand(op->getOperand(itr));
        opOprnds.push_back(node.opName);
        if (argList != "") argList += ", ";
        argList += "strm" + std::to_string(strmVarCount);
        // TODO: Diff between Consumed vs Non-consumed streams
        emit.createStream(oprToStrm, node.opName, strmVarCount,
                          emit.order.empty());
      }
      opCommands.push_back(opOprnds);

      std::vector<std::string> opResults;
      for (int itr = 0; itr < (int)op->getOpResults().size(); itr++) {
        EMITTER_NODE node = emit.ProcessResult(op->getOpResult(itr));
        if (itr == 0) opResults.push_back(node.opName);
        if (argList != "") argList += ", ";
        argList += "strm" + std::to_string(strmVarCount);
        emit.createStream(oprToStrm, node.opName, strmVarCount, false);
      }
      opCommands.push_back(opResults);

      if (llvm::isa<scifrbool::KSOp>(op)) {
        emit.emitKSOp(op, argList);
      } else if (llvm::isa<scifrbool::LinearOp>(op)) {
        emit.emitLinearOp(op, argList);
      } else if (llvm::isa<scifrbool::PBSOp>(op)) {
        emit.emitPBSOp(op, argList);
      } else if (llvm::isa<scifrbool::AndOp>(op)) {
        emit.emitAndOp(op, argList);
      } else if (llvm::isa<scifrbool::NandOp>(op)) {
        emit.emitNandOp(op, argList);
      } else if (llvm::isa<scifrbool::NorOp>(op)) {
        emit.emitNorOp(op, argList);
      } else if (llvm::isa<scifrbool::NotOp>(op)) {
        emit.emitNotOp(op, argList);
      } else if (llvm::isa<scifrbool::OrOp>(op)) {
        emit.emitOrOp(op, argList);
      } else if (llvm::isa<scifrbool::XNorOp>(op)) {
        emit.emitXNorOp(op, argList);
      } else if (llvm::isa<scifrbool::XorOp>(op)) {
        emit.emitXorOp(op, argList);
      }
      opCommands.push_back(emit.commands);
      emit.order.push_back(opCommands);
      emit.commands.clear();

      return WalkResult::advance();
    });

    std::vector<std::string> delOps;
    for (auto itr : emit.inps) {
      if (emit.outs.find((itr)) != emit.outs.end()) {
        delOps.push_back(itr);
      }
    }

    for (auto itr : delOps) {
      emit.inps.erase(itr);
      emit.outs.erase(itr);
      emit.inpStreamNames.erase(itr);
      emit.outStreamNames.erase(itr);
    }

    for (auto itr : emit.inps) {
      emit.nodes[itr].dims = formatDims(emit.nodes[itr].dims);
      for (int ctr = 1; ctr < emit.nodes[itr].outsUsed; ctr++)
        emit.getOrMakeTemporaryStream(itr);
    }

    for (auto itr : emit.outs) {
      emit.nodes[itr].dims = formatDims(emit.nodes[itr].dims);
    }

    emit.EndCode(strmVarCount);
    emit.m_codestr.close();
  }
};  // struct CGGIEmitter

void SCIFRBoolEmitter::createStream(
    std::unordered_map<std::string, std::string>& oprToStrm, std::string opName,
    int& strmVarCount, bool isInput) {
  if (oprToStrm.find(opName) != oprToStrm.end()) {
    // TODO: Check how to create new stream from Consumed input streams
    return;
  } else if (isInput) {
    oprToStrm[opName] = "strm" + std::to_string(strmVarCount);
    m_codestr << R"code(
    StreamLwe<uint64_t> strm)code"
              << strmVarCount
              << R"code( = engineBoolean.newHostToFabricLweStream();)code";
  } else {
    oprToStrm[opName] = "strm" + std::to_string(strmVarCount);
    m_codestr << R"code(
    StreamLwe<uint64_t> strm)code"
              << strmVarCount << R"code( = engineBoolean.newLweStream();)code";
  }
  strmVarCount++;
}

void SCIFRBoolEmitter::emitOp(Operation* op, std::string operationName,
                              std::string argumentList) {
  m_codestr << R"code(
    nResult = engineBoolean.binaryBooleanConsumeAll()code"
            << argumentList << R"code(, BinaryBooleanOp::)code" << operationName
            << R"code(, 1);
    if (nResult != 0) {
        return -1;
    }
)code";
}

void SCIFRBoolEmitter::emitAndOp(Operation* op, std::string argList) {
  emitOp(op, "And", argList);
}
void SCIFRBoolEmitter::emitNandOp(Operation* op, std::string argList) {
  emitOp(op, "Nand", argList);
}
void SCIFRBoolEmitter::emitXorOp(Operation* op, std::string argList) {
  emitOp(op, "Xor", argList);
}
void SCIFRBoolEmitter::emitOrOp(Operation* op, std::string argList) {
  emitOp(op, "Or", argList);
}
void SCIFRBoolEmitter::emitNorOp(Operation* op, std::string argList) {
  emitOp(op, "Nor", argList);
}
void SCIFRBoolEmitter::emitXNorOp(Operation* op, std::string argList) {
  emitOp(op, "Xnor", argList);
}
void SCIFRBoolEmitter::emitNotOp(Operation* op, std::string argList) {
  emitOp(op, "Not", argList);
}
void SCIFRBoolEmitter::emitKSOp(Operation* op, std::string argList) {
  emitOp(op, "KS", argList);
}
void SCIFRBoolEmitter::emitLinearOp(Operation* op, std::string argList) {
  emitOp(op, "Linear", argList);
}
void SCIFRBoolEmitter::emitPBSOp(Operation* op, std::string argList) {
  emitOp(op, "PBS", argList);
}

void SCIFRBoolEmitter::BeginCode() {
  m_codestr << R"code(

#include <concrete_engine.hpp>
#include <concrete_engine_boolean.hpp>
#include <array>

using cornami::fhe::ServerParameters;
using cornami::fhe::concrete::BootstrapKeyStandard;
using cornami::fhe::concrete::KeySwitchKey;
using cornami::fhe::concrete::ConcreteEngine;
using cornami::fhe::concrete::StreamLwe;

using cornami::fhe::concrete::ConcreteEngineBoolean;
using cornami::fhe::concrete::BooleanEncodingType;
using cornami::fhe::concrete::BinaryBooleanOp;

int test_logic_gate(BootstrapKeyStandard<uint64_t>& bsk, KeySwitchKey<uint64_t>& ksk, ServerParameters& serverSetup,
    std::vector<uint64_t>& input1, std::vector<uint64_t>& input2, std::vector<uint64_t>& output) {
    ConcreteEngine<uint64_t>* pEngine = new ConcreteEngine<uint64_t>(bsk, ksk, serverSetup);
    ConcreteEngineBoolean<uint64_t> engineBoolean(pEngine, BooleanEncodingType::NegativeOneOne);
    pEngine = NULL;
    int nResult;
)code";
}

void SCIFRBoolEmitter::EndCode(int strmVarCount) {
  m_codestr << R"code(
    nResult = engineBoolean.m_engine->routeToHost(strm)code"
            << strmVarCount - 1 << R"code();
    if (nResult != 0) {
        return -1;
    }
    nResult = engineBoolean.m_engine->finalizeAndRun();
    if (nResult != 0) {
        return -1;
    }
    uint nLweSize = bsk.m_bskInfo.lweInfo.nLWESize;
    nResult = engineBoolean.m_engine->put(strm0, &input1[0], nLweSize);
    if (nResult != 0) {
        return -1;
    }
    nResult = engineBoolean.m_engine->put(strm1, &input2[0], nLweSize);
    if (nResult != 0) {
        return -1;
    }
    nResult = engineBoolean.m_engine->get(strm)code"
            << strmVarCount - 1 << R"code(, &output[0], nLweSize);
    if (nResult != 0) {
        return -1;
    }
    return 0;
}
)code";
}

EMITTER_NODE SCIFRBoolEmitter::ProcessOperand(mlir::Value val) {
  std::string opName = getValueName(val);
  EMITTER_NODE node;
  if (nodes.find(opName) == nodes.end()) {
    mlir::TensorType mlirTensorType = dyn_cast<mlir::TensorType>(val.getType());
    if (mlirTensorType !=
        NULL) {  // if operand is integer or float need to handle it.
      populateDims(mlirTensorType, node);
    }
    node.opName = opName;
    node.numUsed = 0;
    node.outsUsed = 1;
    inps.insert(opName);
    nodes[opName] = node;
  } else {
    nodes[opName].outsUsed += 1;
  }
  return nodes[opName];
}

EMITTER_NODE SCIFRBoolEmitter::ProcessResult(mlir::Value val) {
  std::string opName = getValueName(val);
  EMITTER_NODE node;
  mlir::TensorType mlirTensorType = dyn_cast<mlir::TensorType>(val.getType());
  if (mlirTensorType !=
      NULL) {  // if operand is integer or float need to handle it.
    populateDims(mlirTensorType, node);
  }
  node.opName = opName;
  outs.insert(opName);
  node.outsUsed = 0;
  node.numUsed = 0;
  nodes[opName] = node;
  return nodes[opName];
}

std::string SCIFRBoolEmitter::getAttrValue(mlir::Attribute attr) {
  std::string opName;
  llvm::raw_string_ostream ss(opName);
  attr.print(ss, false);
  return opName;
}

std::string SCIFRBoolEmitter::getValueName(mlir::Value val) {
  std::string opName;
  llvm::raw_string_ostream ss(opName);
  mlir::OpPrintingFlags pFlags;

  val.printAsOperand(ss, pFlags);
  return opName;
}

// FIXME: get the existing stream if present for the operator at the operand
// input.
std::string SCIFRBoolEmitter::getOrMakeTemporaryStream(Operation* op,
                                                       uint8_t opidx) {
  return getOrMakeTemporaryStream(getValueName(op->getOperand(opidx)));
}

std::string SCIFRBoolEmitter::getOrMakeTemporaryStream(std::string opName) {
  int ind;
  inpStreamCtr++;
  if (inpStreamNames.find(opName) == inpStreamNames.end()) {
    ind = inpStreamNames.size();
    inpStreamNames[opName] = std::vector<VAR_NAMES>();
  } else {
    ind = inpStreamNames[opName].back().fileInd;
  }

  inputCallbacks["outStrm" + std::to_string(inpStreamCtr)] =
      std::make_pair<std::string, std::string>(
          "values_in" + std::to_string(inpStreamCtr),
          "HostInput" + std::to_string(inpStreamCtr));

  inpStreamNames[opName].push_back(createVarNames(
      opName, "outStrm" + std::to_string(inpStreamCtr),
      inputCallbacks["outStrm" + std::to_string(inpStreamCtr)].first,
      "outStrm" + std::to_string(inpStreamCtr),
      inputCallbacks["outStrm" + std::to_string(inpStreamCtr)].second,
      "inputParam" + std::to_string(inpStreamCtr),
      "outStrm" + std::to_string(inpStreamCtr), ind));
  return inpStreamNames[opName].back().StreamName;
}

}  // namespace cornami
}  // namespace mlir
