#ifndef LIB_DIALECT_COMB_CONVERSIONS_COMBTOCGGI_COMBTOCGGI_H_
#define LIB_DIALECT_COMB_CONVERSIONS_COMBTOCGGI_COMBTOCGGI_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::comb {

#define GEN_PASS_DECL
#include "lib/Dialect/Comb/Conversions/CombToCGGI/CombToCGGI.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Comb/Conversions/CombToCGGI/CombToCGGI.h.inc"

}  // namespace mlir::heir::comb

#endif  // LIB_DIALECT_COMB_CONVERSIONS_COMBTOCGGI_COMBTOCGGI_H_
