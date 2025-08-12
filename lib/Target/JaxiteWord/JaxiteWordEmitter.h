#ifndef LIB_TARGET_JAXITEWORD_JAXITEWORDEMITTER_H_
#define LIB_TARGET_JAXITEWORD_JAXITEWORDEMITTER_H_

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
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

  // ciphertext arg.
  std::string CiphertextArg_;

  // A list of modulus to be used for the add operation.
  std::string ModulusListArg_;

  LogicalResult printOperation(ModuleOp moduleOp);
  LogicalResult printOperation(func::FuncOp funcOp);
  LogicalResult printOperation(func::ReturnOp returnOp);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(memref::LoadOp op);
  LogicalResult printOperation(memref::StoreOp op);
  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);

  void emitAssignPrefix(Value result);
};

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITEWORD_JAXITEWORDEMITTER_H_
