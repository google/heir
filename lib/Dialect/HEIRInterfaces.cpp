#include "lib/Dialect/HEIRInterfaces.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

namespace mlir {
namespace heir {

#include "lib/Dialect/HEIROpInterfaces.cpp.inc"
#include "lib/Dialect/HEIRTypeInterfaces.cpp.inc"

void registerOperandAndResultAttrInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, affine::AffineDialect* dialect) {
    affine::AffineForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
}

void registerIncreasesMulDepthOpInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect* dialect) {
    arith::MulIOp::attachInterface<IncreasesMulDepthOpInterface>(*ctx);
    arith::MulFOp::attachInterface<IncreasesMulDepthOpInterface>(*ctx);
  });
}

LogicalResult verifyElementwiseByOperandImpl(
    ElementwiseByOperandOpInterface opInterface) {
  Operation* op = opInterface.getOperation();

  auto typeToShapeStr = [](Type type) {
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
      std::string shapeStr = "(";
      for (auto dim : rankedTensorType.getShape()) {
        shapeStr += std::to_string(dim) + ",";
      }
      shapeStr += ")";
      return shapeStr;
    }
    return std::string("(not statically ranked)");
  };

  std::optional<TensorType> tensorType;
  int64_t operandIndex = 0;
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    auto thisTensorType = dyn_cast<TensorType>(operand.getType());
    if (!thisTensorType)
      // Non-tensor types are acceptable, and need not be specified as mappable
      // or not mappable by the interface.
      continue;

    if (opInterface.operandIsMappable(i)) {
      if (!tensorType) {
        tensorType = thisTensorType;
        operandIndex = i;
        continue;
      }

      if (thisTensorType.getShape() != tensorType->getShape()) {
        return op->emitOpError()
               << "expected all mappable operands to have the same shape, "
               << "but found shape " << typeToShapeStr(*tensorType)
               << " at operand " << operandIndex << " and "
               << typeToShapeStr(thisTensorType) << " at operand " << i;
      }
    }
  }

  for (auto [i, result] : llvm::enumerate(op->getResults())) {
    auto thisTensorType = dyn_cast<TensorType>(result.getType());
    if (tensorType && !thisTensorType)
      return op->emitOpError()
             << "expected all results operands to have the same tensor shape, "
             << "as the mappable input operands, but found shape "
             << typeToShapeStr(*tensorType) << " at operand " << operandIndex
             << " and result " << i << " of non-tensor type "
             << result.getType();

    if (!tensorType && thisTensorType)
      return op->emitOpError()
             << "No operands were tensor typed, but result at index " << i
             << " is a tensor of shape " << typeToShapeStr(thisTensorType);

    if (tensorType && thisTensorType &&
        thisTensorType.getShape() != tensorType->getShape()) {
      return op->emitOpError()
             << "expected all tensor results to have the same shape as "
                "mappable operands, but found shape "
             << typeToShapeStr(*tensorType) << " at operand " << operandIndex
             << " and shape " << typeToShapeStr(thisTensorType) << " at result "
             << i;
    }
  }

  return success();
}

}  // namespace heir
}  // namespace mlir
