#ifndef LIB_UTILS_APINTUTILS_H_
#define LIB_UTILS_APINTUTILS_H_

#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

APInt multiplicativeInverse(const APInt& x, const APInt& modulo);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APINTUTILS_H_
