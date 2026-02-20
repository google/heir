#ifndef SCIFRBOOL_TRANSFORMS_SCIFRBOOLEMITTER_H
#define SCIFRBOOL_TRANSFORMS_SCIFRBOOLEMITTER_H

#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                  // from @llvm-project

namespace mlir {
namespace cornami {

struct VAR_NAMES {
  std::string mlirVarName;    // %1
  std::string StreamVarName;  // MainStream
  std::string vecVarName;     // input
  std::string tsmName;        // outStrmHost
  std::string CBName;         // HostOutput
  std::string paramVarName;   // inputParam
  std::string StreamName;     // main_pstram
  int fileInd;                // 0
  std::vector<int> dims;
  int commandInd;
  void setDims(int nBatch, int nHeight, int nWidth, int nChannels) {
    this->dims = {nBatch, nHeight, nWidth, nChannels};
  }
};

struct EMITTER_NODE {
  std::string opName;
  int outsUsed;
  int numUsed;
  std::vector<int> dims;
  std::vector<uint32_t> vals;
};

static EMITTER_NODE createOpEmitter(std::string name, std::vector<int> dims) {
  EMITTER_NODE op;
  op.opName = name;
  op.outsUsed = 0;
  op.dims = dims;
  return op;
}

static VAR_NAMES createVarNames(std::string mlirName, std::string strmVarName,
                                std::string vecVarName, std::string tsmName,
                                std::string CBName, std::string paramVName,
                                std::string StreamVar, int fInd) {
  VAR_NAMES var;
  var.mlirVarName = mlirName;
  var.StreamVarName = strmVarName;
  var.vecVarName = vecVarName;
  var.CBName = CBName;
  var.tsmName = tsmName;
  var.paramVarName = paramVName;
  var.StreamName = StreamVar;
  var.fileInd = fInd;
  var.dims = {};
  return var;
}

static std::vector<int> formatDims(std::vector<int> dims) {
  std::vector<int> vDims(4, 1);
  if (dims.size() == 3) {
    for (int j = 0; j < (int)dims.size(); j++) vDims[j + 1] = dims[j];
  } else if (dims.size() == 1) {
    vDims[3] = dims[0];
  } else if (dims.size() == 2) {
    vDims[1] = dims[0];
    vDims[2] = dims[1];
  } else {
    for (int j = 0; j < (int)dims.size(); j++) {
      vDims[j] = dims[j];
    }
  }
  return vDims;
}

class SCIFRBoolEmitterPass
    : public PassWrapper<SCIFRBoolEmitterPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "cfair-codegen"; }

  StringRef getDescription() const final {
    return "Codegenerate from CFAIR Dialect (assume module is in CFAIR "
           "Dialect)";
  }

  void runOnOperation() override;
};

class SCIFRBoolEmitter {
 public:
  SCIFRBoolEmitter()
      : m_str(), m_coeffcount(0), inpStreamCtr(0), outStreamCtr(0){};

  // TODO: add header info for TStreams EMU
  void emitKSOp(Operation *op, std::string argList);
  void emitLinearOp(Operation *op, std::string argList);
  void emitPBSOp(Operation *op, std::string argList);
  void emitAndOp(Operation *op, std::string argList);
  void emitNandOp(Operation *op, std::string argList);
  void emitNorOp(Operation *op, std::string argList);
  void emitNotOp(Operation *op, std::string argList);
  void emitOrOp(Operation *op, std::string argList);
  void emitXNorOp(Operation *op, std::string argList);
  void emitXorOp(Operation *op, std::string argList);
  void emitOp(Operation *op, std::string operationName,
              std::string argumentList);
  void createStream(std::unordered_map<std::string, std::string> &oprToStrm,
                    std::string opName, int &strmVarCount, bool isInput);

 protected:
  std::string getOrMakeTemporaryStream(Operation *op, uint8_t opidx);
  std::string getValueName(mlir::Value);
  std::string getAttrValue(mlir::Attribute attr);
  std::string getOrMakeTemporaryStream(std::string opName);
  bool canIgnoreOp(Operation *op) const;

 private:
  std::stringstream m_str;
  std::ofstream m_codestr;
  friend class CFAIREmitterPass;
  friend struct CGGIEmitter;
  void BeginCode();
  void EndCode(int strmVarCount);
  void createLsapOut(std::string fromFAName);
  void createLsapInp(std::string fromFAName, std::string toFAName);
  void createMsapInp(std::string varName, int idx);
  void InputCallback(std::string CBName, std::string tsmName,
                     std::string vectorName);
  void OutputCallback(std::string CBName, std::string tsmName,
                      std::string vectorName);
  void EncodeStreams(VAR_NAMES var);
  EMITTER_NODE ProcessOperand(mlir::Value val);
  EMITTER_NODE ProcessResult(mlir::Value val);
  std::vector<std::string> commands;
  std::vector<std::vector<std::vector<std::string>>> order;
  std::map<std::string, EMITTER_NODE> nodes;
  std::set<std::string> inps;
  std::set<std::string> outs;
  std::map<std::string, std::vector<VAR_NAMES>> inpStreamNames;
  std::map<std::string, std::vector<VAR_NAMES>> outStreamNames;
  std::map<std::string, std::pair<std::string, std::string>> inputCallbacks;
  std::map<std::string, std::pair<std::string, std::string>> outputCallbacks;
  uint32_t m_coeffcount;
  uint32_t inpStreamCtr;
  uint32_t outStreamCtr;
  uint32_t lsapCounts;
  uint32_t longWireCounts;
};

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CGGIEMITTER
#include "lib/Dialect/SCIFRBool/Transforms/SCIFRBoolEmitter.h.inc"

}  // namespace cornami

}  // namespace mlir

#endif /* SCIFRBOOL_TRANSFORMS_SCIFRBOOLEMITTER_H */
