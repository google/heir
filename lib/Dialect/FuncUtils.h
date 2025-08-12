#ifndef LIB_DIALECT_FUNCUTILS_H_
#define LIB_DIALECT_FUNCUTILS_H_

#include <cstdint>

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

namespace mlir {
namespace heir {

/// Copied from MLIR AsmPrinter.cpp with modification.
/// Sanitize the given name such that it can be used as a valid identifier. If
/// the string needs to be modified in any way, the provided buffer is used to
/// store the new copy,
StringRef sanitizeIdentifier(StringRef name, SmallString<16>& buffer,
                             StringRef allowedPunctChars = "_");

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_FUNCUTILS_H_
