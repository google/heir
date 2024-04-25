#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir::heir::polynomial {

/// Returns true if `root` is a primitive nth root modulo `cmod`.
///
/// To test for a 2k-th primitive root, set `n` = 2*k.
bool isPrimitiveNthRootOfUnity(const ::llvm::APInt &root, unsigned n,
                               const ::llvm::APInt &cmod);

}  // namespace mlir::heir::polynomial
