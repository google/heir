#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"

#include "mlir/include/mlir/IR/Types.h"      // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

bool PolynomialType::isCompatibleWith(Type otherRnsBasisType) const {
  auto other = mlir::dyn_cast<PolynomialType>(otherRnsBasisType);
  if (!other) {
    return false;
  }

  // The coefficient moduli may be different, but the polynomial moduli must
  // agree. This is the typical RNS situation where the point is to avoid using
  // big-integer coefficient moduli by converting them to a smaller set of
  // prime moduli.
  return getRing().getIdeal() == other.getRing().getIdeal();
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
