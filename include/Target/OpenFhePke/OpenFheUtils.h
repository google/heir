#ifndef INCLUDE_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
#define INCLUDE_TARGET_OPENFHEPKE_OPENFHEUTILS_H_

#include <string>

#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

/// Convert a type to a string
::mlir::FailureOr<std::string> convertType(::mlir::Type type);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
