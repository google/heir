#ifndef SCIFRBOOL_TRANSFORMS_REPLACEOPWITHSECTION_H
#define SCIFRBOOL_TRANSFORMS_REPLACEOPWITHSECTION_H

#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"        // from @llvm-project

namespace mlir {
namespace cornami {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_REPLACEOPWITHSECTION
#include "lib/Dialect/SCIFRBool/Conversions/ReplaceOpWithSection.h.inc"

}  // namespace cornami
}  // namespace mlir
#endif /* SCIFRBOOL_TRANSFORMS_REPLACEOPWITHSECTION_H */
