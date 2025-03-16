#include "lib/Dialect/HEIRInterfaces.h"

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"      // from @llvm-project

namespace mlir {
namespace heir {
#include "lib/Dialect/HEIRInterfaces.cpp.inc"

bool PolynomialEvalInterface::supportsPolynomial(Attribute polynomialAttr,
                                                 Dialect *dialect) {
  if (auto *handler = getInterfaceFor(dialect))
    return handler->supportsPolynomial(polynomialAttr);
  return false;
}

FailureOr<Value> PolynomialEvalInterface::constructConstant(
    OpBuilder &builder, Location loc, Attribute constantAttr,
    Type evaluatedType, Dialect *dialect) {
  if (auto *handler = getInterfaceFor(dialect))
    return handler->constructConstant(builder, loc, constantAttr,
                                      evaluatedType);
  return failure();
}

void registerOperandAndResultAttrInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, affine::AffineDialect *dialect) {
    affine::AffineForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
