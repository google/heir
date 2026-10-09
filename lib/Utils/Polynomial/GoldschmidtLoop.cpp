#include "lib/Utils/Polynomial/GoldschmidtLoop.h"

#include <cassert>
#include <cstdint>

#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

GoldschmidtLevelSplit splitGoldschmidtLevels(int64_t levels) {
  assert(levels >= 3 && "Goldschmidt inverse requires at least 3 levels");
  int64_t chebLevels = levels / 2;
  return GoldschmidtLevelSplit{(int64_t{1} << chebLevels) - 1,
                               levels - chebLevels - 1};
}

ChebyshevPolynomial goldschmidtInitialApproximation(double lower, double upper,
                                                    int64_t degree) {
  assert(0 < lower && lower < upper &&
         "Goldschmidt inverse requires 0 < lower < upper");
  auto inverse = [](const ::llvm::APFloat& x) {
    return ::llvm::APFloat(1.0 / x.convertToDouble());
  };
  return approximation::caratheodoryFejerApproximation(
      inverse, static_cast<int32_t>(degree), lower, upper);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
