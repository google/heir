#include "lib/Kernel/Utils.h"

#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

Type dagTypeToMLIRType(const DagType& dagType, OpBuilder& builder) {
  return std::visit(
      [&](auto&& arg) -> Type {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, kernel::IntegerType>) {
          return builder.getIntegerType(arg.bitWidth);
        } else if constexpr (std::is_same_v<T, kernel::FloatType>) {
          if (arg.bitWidth == 32) {
            return builder.getF32Type();
          } else if (arg.bitWidth == 64) {
            return builder.getF64Type();
          } else if (arg.bitWidth == 16) {
            return builder.getF16Type();
          } else {
            llvm_unreachable("Unsupported float bit width");
          }
        } else if constexpr (std::is_same_v<T, kernel::IndexType>) {
          return builder.getIndexType();
        } else if constexpr (std::is_same_v<T, kernel::IntTensorType>) {
          auto elementType = builder.getIntegerType(arg.bitWidth);
          return RankedTensorType::get(arg.shape, elementType);
        } else if constexpr (std::is_same_v<T, kernel::FloatTensorType>) {
          mlir::Type elementType;
          if (arg.bitWidth == 32) {
            elementType = builder.getF32Type();
          } else if (arg.bitWidth == 64) {
            elementType = builder.getF64Type();
          } else if (arg.bitWidth == 16) {
            elementType = builder.getF16Type();
          } else {
            llvm_unreachable("Unsupported float bit width");
          }
          return RankedTensorType::get(arg.shape, elementType);
        }
        llvm_unreachable("Unknown DagType variant");
      },
      dagType.type_variant);
}

DagType mlirTypeToDagType(Type type) {
  return llvm::TypeSwitch<Type, DagType>(type)
      .Case<mlir::IntegerType>(
          [&](auto type) { return DagType::integer(type.getWidth()); })
      .Case<mlir::IndexType>([&](auto type) { return DagType::index(); })
      .Case<mlir::FloatType>(
          [&](auto type) { return DagType::floatTy(type.getWidth()); })
      .Case<mlir::RankedTensorType>([&](auto type) {
        std::vector<int64_t> shape(type.getShape());
        int width = type.getElementType().getIntOrFloatBitWidth();
        return llvm::TypeSwitch<Type, DagType>(type.getElementType())
            .template Case<mlir::IntegerType>(
                [&](auto _) { return DagType::intTensor(width, shape); })
            .template Case<mlir::FloatType>(
                [&](auto _) { return DagType::floatTensor(width, shape); })
            .Default([&](auto _) {
              llvm_unreachable("Unsupported element type");
              return DagType();
            });
      });
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
