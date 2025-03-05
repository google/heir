#ifndef LIB_TARGET_LATTIGO_LATTIGOEMITTER_H_
#define LIB_TARGET_LATTIGO_LATTIGOEMITTER_H_

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Utils/Tablegen/InplaceOpInterface.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

void registerTranslateOptions();

/// Translates the given operation to Lattigo
::mlir::LogicalResult translateToLattigo(::mlir::Operation *op,
                                         llvm::raw_ostream &os,
                                         const std::string &packageName);

class LattigoEmitter {
 public:
  LattigoEmitter(raw_ostream &os, SelectVariableNames *variableNames,
                 const std::string &packageName);

  LogicalResult translate(::mlir::Operation &operation);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames *variableNames;

  const std::string &packageName;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(::mlir::func::CallOp op);
  LogicalResult printOperation(::mlir::arith::ConstantOp op);
  LogicalResult printOperation(::mlir::tensor::ExtractOp op);
  LogicalResult printOperation(::mlir::tensor::FromElementsOp op);
  // Lattigo ops
  // RLWE
  LogicalResult printOperation(RLWENewEncryptorOp op);
  LogicalResult printOperation(RLWENewDecryptorOp op);
  LogicalResult printOperation(RLWENewKeyGeneratorOp op);
  LogicalResult printOperation(RLWEGenKeyPairOp op);
  LogicalResult printOperation(RLWEGenRelinearizationKeyOp op);
  LogicalResult printOperation(RLWEGenGaloisKeyOp op);
  LogicalResult printOperation(RLWENewEvaluationKeySetOp op);
  LogicalResult printOperation(RLWEEncryptOp op);
  LogicalResult printOperation(RLWEDecryptOp op);
  LogicalResult printOperation(RLWELevelReduceNewOp op);
  LogicalResult printOperation(RLWELevelReduceOp op);
  // BGV
  LogicalResult printOperation(BGVNewParametersFromLiteralOp op);
  LogicalResult printOperation(BGVNewEncoderOp op);
  LogicalResult printOperation(BGVNewEvaluatorOp op);
  LogicalResult printOperation(BGVNewPlaintextOp op);
  LogicalResult printOperation(BGVEncodeOp op);
  LogicalResult printOperation(BGVDecodeOp op);
  LogicalResult printOperation(BGVAddNewOp op);
  LogicalResult printOperation(BGVSubNewOp op);
  LogicalResult printOperation(BGVMulNewOp op);
  LogicalResult printOperation(BGVRelinearizeNewOp op);
  LogicalResult printOperation(BGVRescaleNewOp op);
  LogicalResult printOperation(BGVRotateColumnsNewOp op);
  LogicalResult printOperation(BGVRotateRowsNewOp op);
  LogicalResult printOperation(BGVAddOp op);
  LogicalResult printOperation(BGVSubOp op);
  LogicalResult printOperation(BGVMulOp op);
  LogicalResult printOperation(BGVRelinearizeOp op);
  LogicalResult printOperation(BGVRescaleOp op);
  LogicalResult printOperation(BGVRotateColumnsOp op);
  LogicalResult printOperation(BGVRotateRowsOp op);
  // CKKS
  LogicalResult printOperation(CKKSNewParametersFromLiteralOp op);
  LogicalResult printOperation(CKKSNewEncoderOp op);
  LogicalResult printOperation(CKKSNewEvaluatorOp op);
  LogicalResult printOperation(CKKSNewPlaintextOp op);
  LogicalResult printOperation(CKKSEncodeOp op);
  LogicalResult printOperation(CKKSDecodeOp op);
  LogicalResult printOperation(CKKSAddNewOp op);
  LogicalResult printOperation(CKKSSubNewOp op);
  LogicalResult printOperation(CKKSMulNewOp op);
  LogicalResult printOperation(CKKSRelinearizeNewOp op);
  LogicalResult printOperation(CKKSRescaleNewOp op);
  LogicalResult printOperation(CKKSRotateNewOp op);
  LogicalResult printOperation(CKKSAddOp op);
  LogicalResult printOperation(CKKSSubOp op);
  LogicalResult printOperation(CKKSMulOp op);
  LogicalResult printOperation(CKKSRelinearizeOp op);
  LogicalResult printOperation(CKKSRescaleOp op);
  LogicalResult printOperation(CKKSRotateOp op);

  // Helpers for above
  void printErrPanic(std::string_view errName);

  LogicalResult printNewMethod(::mlir::Value result,
                               ::mlir::ValueRange operands, std::string_view op,
                               bool err);

  LogicalResult printEvalInplaceMethod(::mlir::Value evaluator,
                                       ::mlir::ValueRange operands,
                                       std::string_view op, bool err);

  LogicalResult printEvalNewMethod(::mlir::ValueRange results,
                                   ::mlir::Value evaluator,
                                   ::mlir::ValueRange operands,
                                   std::string_view op, bool err);

  LogicalResult printEvalNewMethod(::mlir::Value result,
                                   ::mlir::Value evaluator,
                                   ::mlir::ValueRange operands,
                                   std::string_view op, bool err) {
    return printEvalNewMethod(::mlir::ValueRange(result), evaluator, operands,
                              op, err);
  }

  // Canonicalize Debug Port
  bool isDebugPort(::llvm::StringRef debugPortName);
  ::llvm::StringRef canonicalizeDebugPort(::llvm::StringRef debugPortName);

  // find the actual value used for inplace op
  ::mlir::Value getStorageValue(::mlir::Value value) {
    if (auto *op = value.getDefiningOp()) {
      if (auto inplaceOpInterface = mlir::dyn_cast<InplaceOpInterface>(op)) {
        auto inplace =
            op->getOperand(inplaceOpInterface.getInplaceOperandIndex());
        return getStorageValue(inplace);
      }
    }
    return value;
  }

  // helper on name and type
  std::string getName(::mlir::Value value) {
    // special case for 'nil' emission
    if (value == Value()) {
      return "nil";
    }
    // when the value has no uses, we can not assign it a name
    // otherwise GO would complain "declared and not used"
    if (value.use_empty()) {
      return "_";
    }
    return variableNames->getNameForValue(getStorageValue(value));
  }

  std::string getErrName() {
    static int errCount = 0;
    return "err" + std::to_string(errCount++);
  }

  std::string getDebugAttrMapName() {
    static int debugAttrMapCount = 0;
    return "debugAttrMap" + std::to_string(debugAttrMapCount++);
  }

  std::string getCommaSeparatedNames(::mlir::ValueRange values) {
    return commaSeparatedValues(values,
                                [&](Value value) { return getName(value); });
  }

  std::string getCommaSeparatedNamesWithTypes(::mlir::ValueRange values) {
    return commaSeparatedValues(values, [&](Value value) {
      return getName(value) + " " + convertType(value.getType()).value();
    });
  }

  FailureOr<std::string> getCommaSeparatedTypes(::mlir::TypeRange types) {
    return commaSeparatedTypes(types,
                               [&](Type type) { return convertType(type); });
  }

  // Emit an Lattigo type
  FailureOr<std::string> convertType(::mlir::Type type);

  LogicalResult emitType(Type type);
};

void registerToLattigoTranslation(void);

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGOEMITTER_H_
