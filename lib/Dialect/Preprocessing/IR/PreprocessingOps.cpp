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

::mlir::LogicalResult LoadResourceOp::verify() {
  auto shapedType = cast<ShapedType>(getDestination().getType());
  if (!shapedType.hasStaticShape()) {
    return emitOpError() << "destination type " << shapedType
                         << " must have a static shape";
  }
  return ::mlir::success();
}

Speculation::Speculatability LoadResourceOp::getSpeculatability() {
  return isa<TensorType>(getDestination().getType())
             ? Speculation::Speculatable
             : Speculation::NotSpeculatable;
}

LoadResourceOp LoadResourceOp::getForDestination(Value value) {
  for (Operation* user : value.getUsers()) {
    auto loadResourceOp = dyn_cast<LoadResourceOp>(user);
    if (loadResourceOp && loadResourceOp.getDestination() == value)
      return loadResourceOp;
  }
  return {};
}

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
