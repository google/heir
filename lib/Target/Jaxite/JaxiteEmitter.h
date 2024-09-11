#ifndef INCLUDE_TARGET_JAXITE_JAXITEEMITTER_H_
#define INCLUDE_TARGET_JAXITE_JAXITEEMITTER_H_

#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Jaxite/IR/JaxiteOps.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
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
namespace jaxite {

void registerToJaxiteTranslation();

/// Translates the given operation to Jaxire.
::mlir::LogicalResult translateToJaxite(::mlir::Operation *op,
                                        llvm::raw_ostream &os);

class JaxiteEmitter {
 public:
  JaxiteEmitter(raw_ostream &os, SelectVariableNames *variableNames);

  LogicalResult translate(::mlir::Operation &operation);

 private:
  // Output stream to emit to.
  raw_indented_ostream os;

  // Pre-populated analysis selecting unique variable names for all the SSA
  // values.
  SelectVariableNames *variableNames;

  // Server key arg to run ciphertext ops on the server.
  std::string serverKeySetArg_;

  // Params arg to create test polynomials on the server.
  std::string paramsArg_;

  LogicalResult printOperation(ModuleOp moduleOp);
  LogicalResult printOperation(func::FuncOp funcOp);
  LogicalResult printOperation(func::ReturnOp returnOp);
  LogicalResult printOperation(Lut3Op op);
  LogicalResult printOperation(ConstantOp op);
  LogicalResult printOperation(tensor::ExtractOp op);
  LogicalResult printOperation(tensor::FromElementsOp op);
  LogicalResult printOperation(memref::AllocOp op);
  LogicalResult printOperation(memref::LoadOp op);
  LogicalResult printOperation(memref::StoreOp op);
  LogicalResult emitType(Type type);
  FailureOr<std::string> convertType(Type type);

  void emitAssignPrefix(Value result);
};

}  // namespace jaxite
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TARGET_JAXITE_JAXITEEMITTER_H_
