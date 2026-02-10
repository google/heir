#ifndef LIB_DIALECT_LWE_IR_LWETYPES_H_
#define LIB_DIALECT_LWE_IR_LWETYPES_H_

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "mlir/include/mlir/IR/MLIRContext.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWETypes.h.inc"

namespace mlir {
namespace heir {
namespace lwe {

// Return an LWE ciphertext type with a default plaintext modulus of 4 bits and
// ciphertext modulus of 32 bits. The default 4-bit message space from tfhe-rs
// has LWE dimension 742. The messageWidth parameter specifies the width of the
// application data.
LWECiphertextType getDefaultCGGICiphertextType(MLIRContext* ctx,
                                               int plaintextBits);

inline LWEPlaintextType getCorrespondingPlaintextType(
    LWECiphertextType ctType) {
  MLIRContext* ctx = ctType.getContext();
  return LWEPlaintextType::get(
      ctx, PlaintextSpaceAttr::get(ctx, ctType.getCiphertextSpace().getRing(),
                                   ctType.getPlaintextSpace().getEncoding()));
}

// Return the LWECiphertextType resulting from removing one limb (i.e.,
// the result type of a modulus switch or rescale op). Returns a failure
// if the input type does not have enough limbs.
FailureOr<LWECiphertextType> applyModReduce(LWECiphertextType inputType);

// Return the LWECiphertextType resulting from setting the level to a specific
// value.
LWECiphertextType cloneAtLevel(LWECiphertextType inputType, int64_t level);

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
#endif  // LIB_DIALECT_LWE_IR_LWETYPES_H_
