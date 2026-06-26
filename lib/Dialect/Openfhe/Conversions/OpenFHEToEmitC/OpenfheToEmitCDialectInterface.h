#ifndef LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOEMITC_OPENFHETOEMITCDIALECTINTERFACE_H_
#define LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOEMITC_OPENFHETOEMITCDIALECTINTERFACE_H_

#include <optional>

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "mlir/include/mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir::heir::openfhe {

struct OpenfheToEmitCDialectInterface
    : public mlir::ConvertToEmitCPatternInterface {
  OpenfheToEmitCDialectInterface(mlir::Dialect* dialect)
      : mlir::ConvertToEmitCPatternInterface(dialect) {}

  void populateConvertToEmitCConversionPatterns(
      mlir::ConversionTarget& target, mlir::TypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      std::optional<bool> lowerToCpp) const override;
};

inline void registerOpenfheToEmitCInterface(mlir::DialectRegistry& registry) {
  registry.addExtension(+[](mlir::MLIRContext* ctx, OpenfheDialect* dialect) {
    dialect->addInterfaces<OpenfheToEmitCDialectInterface>();
  });
}

}  // namespace mlir::heir::openfhe

#endif  // LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOEMITC_OPENFHETOEMITCDIALECTINTERFACE_H_
