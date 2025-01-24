#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.cpp.inc"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheOps.cpp.inc"

namespace mlir {
namespace heir {
namespace openfhe {

void OpenfheDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Openfhe/IR/OpenfheOps.cpp.inc"
      >();
}

LogicalResult MakePackedPlaintextOp::verify() {
  auto enc = this->getPlaintext().getType().getPlaintextSpace().getEncoding();
  if (!llvm::isa<lwe::FullCRTPackingEncodingAttr>(enc)) {
    return emitOpError("plaintext type should use full_crt_packing_encoding.");
  }
  return success();
}

LogicalResult MakeCKKSPackedPlaintextOp::verify() {
  auto enc = this->getPlaintext().getType().getPlaintextSpace().getEncoding();
  if (!llvm::isa<lwe::InverseCanonicalEncodingAttr>(enc)) {
    return emitOpError("plaintext type should use inverse_canonical_encoding.");
  }
  return success();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
