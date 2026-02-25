#include "lib/Dialect/Lattigo/IR/LattigoOps.h"

#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/IR/OpDefinition.h"   // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

LogicalResult RLWENewEvaluationKeySetOp::verify() {
  if (getKeys().empty()) {
    return emitError("must have at least one key");
  }

  // 0 or 1 relin key + 0 or more galois keys
  int galoisKeyIndex = 0;
  auto firstKey = getKeys()[0];
  if (isa<RLWERelinearizationKeyType>(firstKey.getType())) {
    galoisKeyIndex = 1;
  }

  for (auto key : getKeys().drop_front(galoisKeyIndex)) {
    if (!isa<RLWEGaloisKeyType>(key.getType())) {
      if (isa<RLWERelinearizationKeyType>(key.getType())) {
        return emitError("RLWERelinearizationKey must be the first key");
      }
      return emitError("key must be of type RLWEGaloisKey");
    }
  }
  return success();
}

LogicalResult RLWENewEncryptorOp::verify() {
  auto keyTypeIsPublic =
      mlir::isa<RLWEPublicKeyType>(getEncryptionKey().getType());
  auto encryptorIsPublic = getEncryptor().getType().getPublicKey();
  if (keyTypeIsPublic != encryptorIsPublic) {
    return emitError(
        "encryption key and encryptor must have the same public/secret type");
  }
  return success();
}

LogicalResult BGVRotateColumnsNewOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicShift(),
                                       getStaticShift());
}
LogicalResult BGVRotateColumnsOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicShift(),
                                       getStaticShift());
}
LogicalResult CKKSRotateNewOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicShift(),
                                       getStaticShift());
}
LogicalResult CKKSRotateOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicShift(),
                                       getStaticShift());
}

::mlir::OpFoldResult BGVRotateColumnsNewOp::getRotationIndex() {
  if (getStaticShift()) return getStaticShiftAttr();
  return getDynamicShift();
}

::mlir::OpFoldResult BGVRotateColumnsOp::getRotationIndex() {
  if (getStaticShift()) return getStaticShiftAttr();
  return getDynamicShift();
}

::mlir::OpFoldResult CKKSRotateNewOp::getRotationIndex() {
  if (getStaticShift()) return getStaticShiftAttr();
  return getDynamicShift();
}

::mlir::OpFoldResult CKKSRotateOp::getRotationIndex() {
  if (getStaticShift()) return getStaticShiftAttr();
  return getDynamicShift();
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
