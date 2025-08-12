#include "lib/Dialect/HEIRInterfaces.h"

#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "lib/Transforms/LayoutOptimization/InterfaceImpl.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project

namespace mlir {
namespace heir {

#include "lib/Dialect/HEIRInterfaces.cpp.inc"

namespace {

template <typename OpTy>
struct DoNothingHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          DoNothingHoistingImpl<OpTy>, OpTy> {
  std::vector<::mlir::heir::Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    return {createTrivialHoister(op)};
  }
};

}  // namespace

void registerOperandAndResultAttrInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, affine::AffineDialect* dialect) {
    affine::AffineForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
}

void registerLayoutConversionHoistableInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect* dialect) {
    arith::AddFOp::attachInterface<DoNothingHoistingImpl<arith::AddFOp>>(*ctx);
    arith::AddIOp::attachInterface<DoNothingHoistingImpl<arith::AddIOp>>(*ctx);
    arith::MulFOp::attachInterface<DoNothingHoistingImpl<arith::MulFOp>>(*ctx);
    arith::MulIOp::attachInterface<DoNothingHoistingImpl<arith::MulIOp>>(*ctx);
    arith::SubFOp::attachInterface<DoNothingHoistingImpl<arith::SubFOp>>(*ctx);
    arith::SubIOp::attachInterface<DoNothingHoistingImpl<arith::SubIOp>>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
