#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

using llvm::APInt;

namespace mlir::heir::polynomial {

bool isPrimitiveNthRootOfUnity(const APInt &root, const unsigned n,
                               const APInt &cmod) {
  assert(root.ule(cmod) && "root must be less than cmod");
  // root bitwidth may be 1 less then cmod
  APInt r = APInt(root).zext(cmod.getBitWidth());
  APInt a = root;
  for (size_t k = 1; k < n; k++) {
    if (a.isOne()) return false;
    a = (a * root).urem(cmod);
  }
  return a.isOne();
}

}  // namespace mlir::heir::polynomial
