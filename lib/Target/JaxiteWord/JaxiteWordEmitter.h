#ifndef LIB_TARGET_JAXITEWORD_JAXITEWORDEMITTER_H_
#define LIB_TARGET_JAXITEWORD_JAXITEWORDEMITTER_H_

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
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
namespace jaxiteword {

void registerToJaxiteWordTranslation();

/// Translates the given operation to Jaxire.
::mlir::LogicalResult translateToJaxiteWord(::mlir::Operation* op,
                                            llvm::raw_ostream& os);

class JaxiteWordEmitter {
 public:
  JaxiteWordEmitter(raw_ostream& os, SelectVariableNames* variableNames);

  LogicalResult translate(::mlir::Operation& operation);

 private:
  // Output stream to emit to.
  raw_indented_ostream os;

  // Pre-populated analysis selecting unique variable names for all the SSA
  // values.
  SelectVariableNames* variableNames;

  LogicalResult printOperation(ModuleOp moduleOp);
  LogicalResult printOperation(func::FuncOp funcOp);
  LogicalResult printOperation(func::CallOp op);
  LogicalResult printOperation(func::ReturnOp returnOp);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(SubOp op);
  LogicalResult printOperation(NegateOp op);
  LogicalResult printOperation(SquareOp op);
  LogicalResult printOperation(MulOp op);
  LogicalResult printOperation(MulNoRelinOp op);
  LogicalResult printOperation(ModReduceOp op);
  LogicalResult printOperation(RotOp op);
  LogicalResult printOperation(RelinOp op);

  LogicalResult printOperation(AddPlainOp op);
  LogicalResult printOperation(SubPlainOp op);
  LogicalResult printOperation(MulPlainOp op);
  LogicalResult printOperation(AddInPlaceOp op);
  LogicalResult printOperation(SubInPlaceOp op);

  LogicalResult printOperation(EncodeOp op);
  LogicalResult printOperation(DecodeOp op);
  LogicalResult printOperation(EncryptOp op);
  LogicalResult printOperation(DecryptOp op);

  LogicalResult printOperation(GenParamsOp op);
  LogicalResult printOperation(GenKeyPairOp op);
  LogicalResult printOperation(GenMulKeyOp op);
  LogicalResult printOperation(GenRotKeyOp op);
  LogicalResult printOperation(ProgramInitializationOp op);

  // Tensor ops
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(tensor::EmptyOp op);
  LogicalResult printOperation(tensor::InsertOp op);
  LogicalResult printOperation(tensor::ExtractSliceOp op);
  LogicalResult printOperation(tensor::InsertSliceOp op);

  // SCF ops
  LogicalResult printOperation(scf::ForOp op);
  LogicalResult printOperation(scf::IfOp op);
  LogicalResult printOperation(scf::YieldOp op);

  // Arith ops (constants and loop indices)
  LogicalResult printOperation(arith::ConstantOp op);
  LogicalResult printOperation(arith::IndexCastOp op);
  LogicalResult printOperation(arith::AddIOp op);
  LogicalResult printOperation(arith::SubIOp op);
  LogicalResult printOperation(arith::MulIOp op);
  LogicalResult printOperation(arith::DivSIOp op);
  LogicalResult printOperation(arith::RemSIOp op);
  LogicalResult printOperation(arith::CmpIOp op);
  LogicalResult printOperation(arith::SelectOp op);
  LogicalResult printOperation(arith::ExtSIOp op);
  LogicalResult printOperation(arith::ExtUIOp op);
  LogicalResult printOperation(arith::TruncIOp op);

  // Memref ops
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(memref::LoadOp op);
  LogicalResult printOperation(memref::StoreOp op);

  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);

  void emitAssignPrefix(Value result);
  void emitAssignCiphertext(StringRef targetName, StringRef sourceName);
  void emitNormalizeCiphertext(StringRef resultName, StringRef ctxName,
                               StringRef sourceName, StringRef levelExpr = "");
  void emitModularAdd(StringRef resultName, StringRef ctxName,
                      StringRef lhsName, StringRef rhsName);
  void emitModularReduce(StringRef targetName);

  LogicalResult printMulOpHelper(
      Value result, Value lhs, Value rhs, Value ctx, Operation* op,
      llvm::function_ref<void(StringRef, StringRef, StringRef, StringRef,
                              StringRef)>
          callback);
};

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITEWORD_JAXITEWORDEMITTER_H_
