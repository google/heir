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
// FIXME: add reference
LWECiphertextType getDefaultCGGICiphertextType(MLIRContext* ctx,
                                               int messageWidth,
                                               int plaintextBits = 3);

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_IR_LWETYPES_H_
