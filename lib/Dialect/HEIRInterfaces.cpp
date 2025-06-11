#include "lib/Dialect/HEIRInterfaces.h"

#include "lib/Transforms/LayoutOptimization/InterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project

namespace mlir {
namespace heir {

#include "lib/Dialect/HEIRInterfaces.cpp.inc"

void registerOperandAndResultAttrInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, affine::AffineDialect *dialect) {
    affine::AffineForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
}

void registerLayoutConversionHoistableInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::AddIOp::attachInterface<DoNothingHoistingImpl<arith::AddIOp>>(*ctx);
    arith::AddFOp::attachInterface<DoNothingHoistingImpl<arith::AddFOp>>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
