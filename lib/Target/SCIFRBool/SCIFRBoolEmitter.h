#ifndef SCIFRBOOL_TARGET_SCIFRBOOLEMITTER_H
#define SCIFRBOOL_TARGET_SCIFRBOOLEMITTER_H

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project

namespace mlir {
namespace cornami {
namespace scifrbool {

void registerToSCIFRBoolTranslation();
void registerTranslateOptions();

// Translates the given operation to SCIFRBool.
::mlir::LogicalResult translateToSCIFRBool(::mlir::Operation* op,
                                           llvm::raw_ostream& os,
                                           std::string vizdir);

class SCIFRBoolEmitter {
 public:
  SCIFRBoolEmitter(raw_ostream& os, std::string vizdir);
  LogicalResult translate(Operation& operation);
  void autoGenComment();
  void beginCode();

 private:
  // Output stream to emit to.
  std::string m_visualization_dir;
  raw_indented_ostream m_os;
  llvm::DenseMap<Value, int> m_variableNames;
  int m_variableCount = 0;
  int m_sectionCount = 0;
  std::string m_prefix{"v"};
  uint32_t m_strmVarCount = 0;
  std::unordered_map<std::string, std::vector<std::string>> m_oprToStrm;
  std::unordered_set<std::string> m_isConsumed;
  std::vector<Value> m_argList;
  std::string m_returnStr, m_fromElementsStr;
  std::unordered_set<std::string> m_scifrTypes = {
      "BootstrapKeyStandard<uint64_t>", "KeySwitchKey<uint64_t>",
      "ServerParameters"};
  std::unordered_map<std::string, std::vector<std::string>> m_extractedValues;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(AndOp op);
  LogicalResult printOperation(NandOp op);
  LogicalResult printOperation(OrOp op);
  LogicalResult printOperation(NorOp op);
  LogicalResult printOperation(NotOp op);
  LogicalResult printOperation(XorOp op);
  LogicalResult printOperation(XNorOp op);
  LogicalResult printOperation(SectionOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(::mlir::arith::ConstantOp op);
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(memref::LoadOp op);
  LogicalResult printOperation(memref::StoreOp op);

  // Helpers for above
  LogicalResult printSksMethod(::mlir::ValueRange operands,
                               ::mlir::Value result,
                               std::string_view operationName);

  // Emit a SCIFRBool type
  LogicalResult emitType(Type type);
  LogicalResult emitTypedAssignPrefix(Value result);
  FailureOr<std::string> convertType(Type type);

  void createStream(std::string opName);

  void createVariableNames(Operation* op) {
    op->walk<WalkOrder::PreOrder>([&](Operation* opr) {
      for (Value oprnd : opr->getOperands()) {
        m_variableNames.try_emplace(oprnd, m_variableCount++);
      }
      for (Value result : opr->getResults()) {
        m_variableNames.try_emplace(result, m_variableCount++);
      }
      for (Region& region : opr->getRegions()) {
        for (Block& block : region) {
          for (Value arg : block.getArguments()) {
            m_variableNames.try_emplace(arg, m_variableCount++);
          }
        }
      }
    });
  }

  // Return the name assigned to the given value
  std::string getNameForValue(Value value) const {
    assert(m_variableNames.contains(value));
    return m_prefix + std::to_string(m_variableNames.lookup(value));
  }

  // Return the unique integer assigned to a given value.
  int getIntForValue(Value value) const {
    assert(m_variableNames.contains(value));
    return m_variableNames.lookup(value);
  }

  void errorCheck();

  LogicalResult canEmitFuncForSCIFRBool(func::FuncOp& funcOp) {
    WalkResult failIfInterrupted = funcOp.walk([&](Operation* op) {
      return TypeSwitch<Operation*, WalkResult>(op)
          .Case<ModuleOp, func::FuncOp, func::ReturnOp, arith::ConstantOp,
                tensor::ExtractOp, tensor::FromElementsOp, memref::AllocOp,
                memref::LoadOp, memref::StoreOp, AndOp, NandOp, OrOp, NorOp,
                NotOp, XorOp, XNorOp, SectionOp>(
              [&](auto op) { return WalkResult::advance(); })
          .Default([&](Operation* op) {
            llvm::errs()
                << "// Skipping function " << funcOp.getName()
                << " which cannot be emitted because it has an unsupported op: "
                << *op << "\n";
            return WalkResult::interrupt();
          });
    });

    if (failIfInterrupted.wasInterrupted()) return failure();
    return success();
  }
};

}  // namespace scifrbool
}  // namespace cornami
}  // namespace mlir

#endif /* SCIFRBOOL_TARGET_SCIFRBOOLEMITTER_H */
