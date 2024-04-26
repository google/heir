#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_

#include <string>

#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

/// Convert a type to a string.
::mlir::FailureOr<std::string> convertType(::mlir::Type type);

/// Find the CryptoContext SSA value in the input operation's parent func
/// arguments.
::mlir::FailureOr<::mlir::Value> getContextualCryptoContext(
    ::mlir::Operation *op);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
