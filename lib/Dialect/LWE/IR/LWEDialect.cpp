#include "include/Dialect/LWE/IR/LWEDialect.h"

#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWEOps.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions
#include "include/Dialect/LWE/IR/LWEDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/LWE/IR/LWEAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "include/Dialect/LWE/IR/LWETypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/LWE/IR/LWEOps.cpp.inc"

namespace mlir {
namespace heir {
namespace lwe {

//===----------------------------------------------------------------------===//
// LWE dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void LWEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "include/Dialect/LWE/IR/LWEAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/LWE/IR/LWETypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/LWE/IR/LWEOps.cpp.inc"
      >();
}

LogicalResult LWEEncodingSchemeAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    unsigned int plaintextBitwidth, unsigned int paddingBitwidth,
    unsigned int cleartextBitwidth) {
  // It may be worth adding some sort of warning notification if the attribute
  // allocates no bits for noise.
  if (plaintextBitwidth < paddingBitwidth + cleartextBitwidth)
    return emitError() << "Attribute's designated plaintext bitwidth ("
                       << plaintextBitwidth
                       << ") is too small to store both the cleartext ("
                       << cleartextBitwidth << ") and padding ("
                       << paddingBitwidth << ")";
  return success();
}

LogicalResult ModulusSwitchOp::verify() {
  uint64_t attrFromLogModulus = getFromLogModulus().getZExtValue();
  uint64_t inputFromLogModulus =
      getInput().getType().getEncodingScheme().getPlaintextBitwidth();
  uint64_t attrToLogModulus = getToLogModulus().getZExtValue();
  uint64_t inputToLogModulus =
      getOutput().getType().getEncodingScheme().getPlaintextBitwidth();

  if (attrFromLogModulus != inputFromLogModulus)
    return emitOpError() << "Modulus switch input has plaintext bitwidth "
                         << inputFromLogModulus
                         << " but the from_log_modulus attr is "
                         << attrFromLogModulus;

  if (attrToLogModulus != inputToLogModulus)
    return emitOpError() << "Modulus switch output has plaintext bitwidth "
                         << inputToLogModulus
                         << " but the to_log_modulus attr is "
                         << attrToLogModulus;
  return success();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
