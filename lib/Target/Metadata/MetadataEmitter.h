#ifndef LIB_TARGET_METADATA_METADATAEMITTER_H_
#define LIB_TARGET_METADATA_METADATAEMITTER_H_

#include "llvm/include/llvm/Support/JSON.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

void registerMetadataEmitter();

/// Emits metadata for the given operation. Supports only modules and funcs as
/// top level inputs.
mlir::LogicalResult emitMetadata(mlir::Operation *op, llvm::raw_ostream &os);

class MetadataEmitter {
 public:
  MetadataEmitter() = default;

  FailureOr<llvm::json::Object> translate(mlir::Operation &operation);

 private:
  FailureOr<llvm::json::Object> emitOperation(mlir::ModuleOp op);
  FailureOr<llvm::json::Object> emitOperation(mlir::func::FuncOp op);

  FailureOr<llvm::json::Object> typeAsJson(MemRefType &ty);
  FailureOr<llvm::json::Object> typeAsJson(IntegerType &ty);
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_METADATA_METADATAEMITTER_H_
