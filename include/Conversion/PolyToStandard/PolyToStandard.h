#ifndef INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_H_
#define INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

#define GEN_PASS_DECL
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

}  // namespace poly
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_H_
