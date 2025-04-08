#include "lib/Dialect/HEIRInterfaces.h"

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
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

Value PolynomialEvalInterface::constructConstant(OpBuilder &builder,
                                                 Location loc,
                                                 Attribute constantAttr,
                                                 Type evaluatedType,
                                                 Dialect *dialect) {
  if (auto *handler = getInterfaceFor(dialect))
    return handler->constructConstant(builder, loc, constantAttr,
                                      evaluatedType);
  llvm_unreachable("unsupported dialect");
  return Value();
}

Value PolynomialEvalInterface::constructMul(OpBuilder &builder, Location loc,
                                            Value lhs, Value rhs,
                                            Dialect *dialect) {
  if (auto *handler = getInterfaceFor(dialect))
    return handler->constructMul(builder, loc, lhs, rhs);
  llvm_unreachable("unsupported dialect");
  return Value();
}

Value PolynomialEvalInterface::constructAdd(OpBuilder &builder, Location loc,
                                            Value lhs, Value rhs,
                                            Dialect *dialect) {
  if (auto *handler = getInterfaceFor(dialect))
    return handler->constructAdd(builder, loc, lhs, rhs);
  llvm_unreachable("unsupported dialect");
  return Value();
}

// Interface implementations of upstream dialects
namespace {
struct ArithPolynomialEvalInterface : public DialectPolynomialEvalInterface {
  using DialectPolynomialEvalInterface::DialectPolynomialEvalInterface;

  bool supportsPolynomial(Attribute polynomialAttr) const final {
    return isa<::mlir::heir::polynomial::IntPolynomialAttr,
               ::mlir::heir::polynomial::TypedIntPolynomialAttr,
               ::mlir::heir::polynomial::FloatPolynomialAttr,
               ::mlir::heir::polynomial::TypedFloatPolynomialAttr>(
        polynomialAttr);
  }

  Value constructConstant(OpBuilder &builder, Location loc,
                          Attribute constantAttr,
                          Type evaluatedType) const final {
    TypedAttr typedAttr = dyn_cast<TypedAttr>(constantAttr);
    return builder
        .create<::mlir::arith::ConstantOp>(loc, evaluatedType, typedAttr)
        .getResult();
  }

  Value constructMul(OpBuilder &builder, Location loc, Value lhs,
                     Value rhs) const final {
    if (lhs.getType().isFloat() && rhs.getType().isFloat()) {
      return builder.create<::mlir::arith::MulFOp>(loc, lhs, rhs).getResult();
    }

    return builder.create<::mlir::arith::MulIOp>(loc, lhs, rhs).getResult();
  }

  Value constructAdd(OpBuilder &builder, Location loc, Value lhs,
                     Value rhs) const final {
    if (lhs.getType().isFloat() && rhs.getType().isFloat()) {
      return builder.create<::mlir::arith::AddFOp>(loc, lhs, rhs).getResult();
    }

    return builder.create<::mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
  }
};
}  // namespace

void registerOperandAndResultAttrInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, affine::AffineDialect *dialect) {
    affine::AffineForOp::attachInterface<OperandAndResultAttrInterface>(*ctx);
  });
}

void registerPolynomialEvalInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    dialect->addInterfaces<ArithPolynomialEvalInterface>();
  });
}

}  // namespace heir
}  // namespace mlir
