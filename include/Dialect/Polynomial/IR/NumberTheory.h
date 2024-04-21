#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir::heir::polynomial {

/// Returns true if `root` is a primitive nth root modulo `cmod`.
bool isPrimitiveNthRootOfUnity(const ::llvm::APInt &root, unsigned n,
                               const ::llvm::APInt &cmod);

/// Returns true if `root` is a primitive 2nth root modulo `cmod`.
inline bool isPrimitive2nthRootOfUnity(const ::llvm::APInt &root, unsigned n,
                                       const ::llvm::APInt &cmod) {
  return isPrimitiveNthRootOfUnity(root, 2 * n, cmod);
}

}  // namespace mlir::heir::polynomial
