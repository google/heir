#include "lib/Dialect/HEIRInterfaces.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

#include "lib/Dialect/HEIROpInterfaces.cpp.inc"
#include "lib/Dialect/HEIRTypeInterfaces.cpp.inc"

using arith::AddFOp;
using arith::AddIOp;
using arith::MulFOp;
using arith::MulIOp;
using arith::SubFOp;
using arith::SubIOp;

void registerOperandAndResultAttrInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, affine::AffineDialect* dialect) {
    affine::AffineForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, scf::SCFDialect* dialect) {
    scf::ForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, func::FuncDialect* dialect) {
    func::CallOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
}

void registerIncreasesMulDepthOpInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect* dialect) {
    MulIOp::attachInterface<IncreasesMulDepthOpInterface>(*ctx);
    MulFOp::attachInterface<IncreasesMulDepthOpInterface>(*ctx);
  });
}

namespace {
template <typename OpTy>
struct AnyOperandMayBePlaintextImpl
    : public PlaintextOperandInterface::ExternalModel<
          AnyOperandMayBePlaintextImpl<OpTy>, OpTy> {
  SmallVector<unsigned> maybePlaintextOperands(Operation* op) const {
    return {0, 1};
  }
};
}  // namespace

void registerPlaintextOperandInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect* dialect) {
    MulIOp::attachInterface<AnyOperandMayBePlaintextImpl<MulIOp>>(*ctx);
    MulFOp::attachInterface<AnyOperandMayBePlaintextImpl<MulFOp>>(*ctx);
    AddIOp::attachInterface<AnyOperandMayBePlaintextImpl<AddIOp>>(*ctx);
    AddFOp::attachInterface<AnyOperandMayBePlaintextImpl<AddFOp>>(*ctx);
    SubIOp::attachInterface<AnyOperandMayBePlaintextImpl<SubIOp>>(*ctx);
    SubFOp::attachInterface<AnyOperandMayBePlaintextImpl<SubFOp>>(*ctx);
  });
}

LogicalResult verifyElementwiseByOperandImpl(
    ElementwiseByOperandOpInterface opInterface) {
  Operation* op = opInterface.getOperation();

  auto typeToShapeStr = [](ArrayRef<int64_t> shape) {
    return "(" +
           llvm::join(
               llvm::map_range(shape,
                               [](int64_t dim) { return std::to_string(dim); }),
               ", ") +
           ")";
  };

  std::optional<SmallVector<int64_t>> mappedShape;
  int64_t firstMappableOperandIndex = -1;

  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    auto thisTensorType = dyn_cast<TensorType>(operand.getType());
    if (!thisTensorType) continue;

    if (opInterface.operandIsMappable(i)) {
      SmallVector<int64_t> thisMappedShape;
      for (int dim : opInterface.mappedDimensionsForOperand(i)) {
        thisMappedShape.push_back(thisTensorType.getDimSize(dim));
      }

      if (!mappedShape) {
        mappedShape = thisMappedShape;
        firstMappableOperandIndex = i;
        continue;
      }

      if (thisMappedShape != *mappedShape) {
        return op->emitOpError()
               << "expected all mappable operands to have the same mapped "
                  "shape, but found mapped shape "
               << typeToShapeStr(*mappedShape) << " at operand "
               << firstMappableOperandIndex << " and "
               << typeToShapeStr(thisMappedShape) << " at operand " << i;
      }
    }
  }

  for (auto [i, result] : llvm::enumerate(op->getResults())) {
    auto thisTensorType = dyn_cast<TensorType>(result.getType());
    if (mappedShape && !mappedShape->empty() && !thisTensorType)
      return op->emitOpError()
             << "expected all results to be tensors with shape "
             << typeToShapeStr(*mappedShape)
             << " due to mappable operands, but result " << i
             << " is of non-tensor type " << result.getType();

    if (!mappedShape && thisTensorType)
      return op->emitOpError()
             << "No operands were mappable, but result at index " << i
             << " is a tensor of shape "
             << typeToShapeStr(thisTensorType.getShape());

    if (mappedShape && thisTensorType &&
        thisTensorType.getShape() != ArrayRef<int64_t>(*mappedShape)) {
      return op->emitOpError()
             << "expected all tensor results to have the same shape as "
                "mappable operands, but found shape "
             << typeToShapeStr(*mappedShape) << " at operand "
             << firstMappableOperandIndex << " and shape "
             << typeToShapeStr(thisTensorType.getShape()) << " at result " << i;
    }
  }

  return success();
}

}  // namespace heir
}  // namespace mlir
