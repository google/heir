#ifndef LIB_TRANSFORMS_POLYNOMIALCOEFFFOLDING_POLYNOMIALCOEFFFOLDING_H_
#define LIB_TRANSFORMS_POLYNOMIALCOEFFFOLDING_POLYNOMIALCOEFFFOLDING_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/PolynomialCoeffFolding/PolynomialCoeffFolding.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/PolynomialCoeffFolding/PolynomialCoeffFolding.h.inc"

std::unique_ptr<Pass> createPolynomialCoeffFoldingPass();

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_POLYNOMIALCOEFFFOLDING_POLYNOMIALCOEFFFOLDING_H_
