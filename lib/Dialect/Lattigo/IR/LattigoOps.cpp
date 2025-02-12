#include "lib/Dialect/Lattigo/IR/LattigoOps.h"

#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

LogicalResult BGVEncodeOp::verify() {
  if (!isa<RankedTensorType>(getValue().getType())) {
    return emitError("value must be a ranked tensor");
  }
  return success();
}

LogicalResult BGVDecodeOp::verify() {
  if (getValue().getType() != getDecoded().getType()) {
    return emitError("value and decoded types must match");
  }
  if (!isa<RankedTensorType>(getDecoded().getType())) {
    return emitError("decoded must be a ranked tensor");
  }
  return success();
}

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

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
