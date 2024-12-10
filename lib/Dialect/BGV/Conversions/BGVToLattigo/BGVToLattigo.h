#ifndef LIB_DIALECT_BGV_CONVERSIONS_BGVTOLATTIGO_BGVTOLATTIGO_H_
#define LIB_DIALECT_BGV_CONVERSIONS_BGVTOLATTIGO_BGVTOLATTIGO_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DECL
#include "lib/Dialect/BGV/Conversions/BGVToLattigo/BGVToLattigo.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/BGV/Conversions/BGVToLattigo/BGVToLattigo.h.inc"

}  // namespace mlir::heir::bgv

#endif  // LIB_DIALECT_BGV_CONVERSIONS_BGVTOLATTIGO_BGVTOLATTIGO_H_
