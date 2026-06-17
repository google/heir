#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"

#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.cpp.inc"

namespace mlir {
namespace heir {
namespace preprocessing {

::mlir::LogicalResult StoreOp::verify() {
  if (getValue().getType() != getElementType()) {
    return emitOpError() << "stored value type " << getValue().getType()
                         << " does not match op element type "
                         << getElementType();
  }

  auto storageType = cast<PreprocessingStorageType>(getStorage().getType());
  if (!llvm::is_contained(storageType.getElementTypes(), getElementType())) {
    return emitOpError() << "op element type " << getElementType()
                         << " is not in storage element types";
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadOp::verify() {
  if (getResult().getType() != getElementType()) {
    return emitOpError() << "loaded value type " << getResult().getType()
                         << " does not match op element type "
                         << getElementType();
  }

  auto storageType = cast<PreprocessingStorageType>(getStorage().getType());
  if (!llvm::is_contained(storageType.getElementTypes(), getElementType())) {
    return emitOpError() << "op element type " << getElementType()
                         << " is not in storage element types";
  }
  return ::mlir::success();
}

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
