#ifndef LIB_DIALECT_SCIFRBOOL_CONVERSIONS_CGGITOSCIFRBOOL_H_
#define LIB_DIALECT_SCIFRBOOL_CONVERSIONS_CGGITOSCIFRBOOL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace cornami {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CGGITOSCIFRBOOL
#include "lib/Dialect/CGGI/Conversions/CGGIToSCIFRBool/CGGIToSCIFRBool.h.inc"

}  // namespace cornami
}  // namespace mlir

#endif  // LIB_DIALECT_SCIFRBOOL_CONVERSIONS_CGGITOSCIFRBOOL_H_
