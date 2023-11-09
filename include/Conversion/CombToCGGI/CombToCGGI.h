#ifndef INCLUDE_CONVERSION_COMBTOCGGI_COMBTOCGGI_H_
#define INCLUDE_CONVERSION_COMBTOCGGI_COMBTOCGGI_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::comb {

#define GEN_PASS_DECL
#include "include/Conversion/CombToCGGI/CombToCGGI.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Conversion/CombToCGGI/CombToCGGI.h.inc"

}  // namespace mlir::heir::comb

#endif  // INCLUDE_CONVERSION_COMBTOCGGI_COMBTOCGGI_H_
