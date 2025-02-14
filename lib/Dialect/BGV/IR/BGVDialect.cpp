#include "lib/Dialect/BGV/IR/BGVDialect.h"

#include <optional>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/FHEHelpers.h"
#include "mlir/include/mlir/IR/Location.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

// Generated definitions
#include "lib/Dialect/BGV/IR/BGVDialect.cpp.inc"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/BGV/IR/BGVAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// BGV dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void BGVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/BGV/IR/BGVAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/BGV/IR/BGVOps.cpp.inc"
      >();
}

LogicalResult MulOp::verify() { return verifyMulOp(this); }

LogicalResult RotateOp::verify() { return verifyRotateOp(this); }

LogicalResult RelinearizeOp::verify() { return verifyRelinearizeOp(this); }

LogicalResult ModulusSwitchOp::verify() {
  return verifyModulusSwitchOrRescaleOp(this);
}

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, AddOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult SubOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, SubOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, MulOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RelinearizeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, RelinearizeOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferRelinearizeOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult ExtractOp::verify() { return verifyExtractOp(this); }

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
