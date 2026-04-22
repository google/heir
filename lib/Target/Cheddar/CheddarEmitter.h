#ifndef LIB_TARGET_CHEDDAR_CHEDDAREMITTER_H_
#define LIB_TARGET_CHEDDAR_CHEDDAREMITTER_H_

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

/// Emits C++ code targeting the CHEDDAR GPU FHE library API.
class CheddarEmitter {
 public:
  CheddarEmitter(raw_ostream &os, SelectVariableNames *variableNames,
                 bool use64Bit = true, const std::string &paramsJsonPath = "");

  LogicalResult translate(Operation &operation);

  void emitPrelude(raw_ostream &os) const;

 private:
  raw_indented_ostream os;
  SelectVariableNames *variableNames;
  bool use64Bit;
  std::string paramsJsonPath;

  // Track which include groups are needed
  bool needsExtensionIncludes = false;
  bool needsJsonIncludes = false;

  // Counter for generating unique temporary variable names (e.g., for copies
  // of multi-use move-only values).
  int tempVarCounter = 0;

  // Module-level scheme params (read from cheddar.* attrs)
  int64_t logDefaultScale = -1;
  int64_t logN = 0;
  std::vector<int64_t> qPrimes;
  std::vector<int64_t> pPrimes;
  std::set<int64_t> rotationDistances;

  // Per-op printers
  LogicalResult printOperation(ModuleOp op);
  LogicalResult printOperation(func::FuncOp op);
  LogicalResult printOperation(func::ReturnOp op);
  LogicalResult printOperation(func::CallOp op);

  // Cheddar dialect ops
  LogicalResult printOperation(CreateContextOp op);
  LogicalResult printOperation(CreateUserInterfaceOp op);
  LogicalResult printOperation(GetEncoderOp op);
  LogicalResult printOperation(GetEvkMapOp op);
  LogicalResult printOperation(GetMultKeyOp op);
  LogicalResult printOperation(GetRotKeyOp op);
  LogicalResult printOperation(GetConjKeyOp op);
  LogicalResult printOperation(PrepareRotKeyOp op);

  LogicalResult printOperation(EncodeOp op);
  LogicalResult printOperation(EncodeConstantOp op);
  LogicalResult printOperation(DecodeOp op);
  LogicalResult printOperation(EncryptOp op);
  LogicalResult printOperation(DecryptOp op);

  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(SubOp op);
  LogicalResult printOperation(MultOp op);
  LogicalResult printOperation(NegOp op);

  LogicalResult printOperation(AddPlainOp op);
  LogicalResult printOperation(SubPlainOp op);
  LogicalResult printOperation(MultPlainOp op);
  LogicalResult printOperation(AddConstOp op);
  LogicalResult printOperation(MultConstOp op);

  LogicalResult printOperation(RescaleOp op);
  LogicalResult printOperation(LevelDownOp op);
  LogicalResult printOperation(RelinearizeOp op);
  LogicalResult printOperation(RelinearizeRescaleOp op);

  LogicalResult printOperation(HMultOp op);
  LogicalResult printOperation(HRotOp op);
  LogicalResult printOperation(HRotAddOp op);
  LogicalResult printOperation(HConjOp op);
  LogicalResult printOperation(HConjAddOp op);
  LogicalResult printOperation(MadUnsafeOp op);

  LogicalResult printOperation(BootOp op);
  LogicalResult printOperation(LinearTransformOp op);
  LogicalResult printOperation(EvalPolyOp op);

  // Arith dialect ops
  LogicalResult printOperation(arith::ConstantOp op);
  LogicalResult printOperation(arith::AddIOp op);
  LogicalResult printOperation(arith::MulIOp op);
  LogicalResult printOperation(arith::SubIOp op);
  LogicalResult printOperation(arith::FloorDivSIOp op);
  LogicalResult printOperation(arith::RemSIOp op);
  LogicalResult printOperation(arith::CmpIOp op);
  LogicalResult printOperation(arith::IndexCastOp op);

  // SCF ops
  LogicalResult printOperation(scf::ForOp op);
  LogicalResult printOperation(scf::IfOp op);
  LogicalResult printOperation(scf::YieldOp op);

  // Tensor ops
  LogicalResult printOperation(tensor::EmptyOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::InsertOp op);
  LogicalResult printOperation(tensor::SplatOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(tensor::ExpandShapeOp op);
  LogicalResult printOperation(tensor::ExtractSliceOp op);
  LogicalResult printOperation(tensor::InsertSliceOp op);

 public:
  // Type conversion (public for header emission)
  // asArg=true emits const-ref for move-only CHEDDAR types (function params)
  // asArg=false emits by-value (return types, local variables)
  FailureOr<std::string> convertType(Type type, bool asArg = false);

 private:
  std::string getName(Value value);
  std::string getContextName(Operation *op);
  void emitScaleMismatchDebugCheck(StringRef opKind, StringRef resultName,
                                   Value lhs, Value rhs);
  LogicalResult emitVectorDeepCopy(Operation *op, StringRef destName,
                                   StringRef srcName, Type elemType,
                                   Type tensorType);
};

/// Free functions for translation registration

LogicalResult translateToCheddar(Operation *op, llvm::raw_ostream &os,
                                 bool use64Bit, const std::string &paramsJson);

void registerToCheddarTranslation();
void registerToCheddarHeaderTranslation();
void registerCheddarTranslateOptions();

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_CHEDDAREMITTER_H_
