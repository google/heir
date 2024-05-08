#include "lib/Dialect/PolyExt/IR/PolyExtDialect.h"

#include "lib/Dialect/PolyExt/IR/PolyExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

// Generated definitions.
#include "lib/Dialect/PolyExt/IR/PolyExtDialect.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/PolyExt/IR/PolyExtOps.cpp.inc"

namespace mlir::heir::poly_ext {

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void PolyExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/PolyExt/IR/PolyExtOps.cpp.inc"
      >();
}

polynomial::PolynomialType getPolynomialType(Type t) {
  if (auto tTnsr = mlir::dyn_cast<RankedTensorType>(t)) {
    return mlir::cast<::mlir::polynomial::PolynomialType>(
        tTnsr.getElementType());
  }
  return mlir::cast<::mlir::polynomial::PolynomialType>(t);
}

LogicalResult CModSwitchOp::verify() {
  auto xRing = getPolynomialType(getX().getType()).getRing();
  auto outRing = getPolynomialType(getOutput().getType()).getRing();
  auto outRingCmod = outRing.getCoefficientModulus().getValue();
  auto xRingCmod = xRing.getCoefficientModulus().getValue();

  if (xRing.getPolynomialModulus() != outRing.getPolynomialModulus()) {
    return emitOpError() << "input and output rings ideals must be the same";
  }

  if (xRingCmod.getBitWidth() != outRingCmod.getBitWidth()) {
    return emitOpError()
           << "input ring cmod and output ring cmod's have different bit "
              "widths; "
              "consider annotating the types with `: i64` or similar, or using "
              "the relevant builder on Ring_Attr";
  }

  if (xRingCmod.ule(outRingCmod)) {
    return emitOpError()
           << "input ring cmod must be larger than output ring cmod";
  }

  APInt congMod = getCongruenceModulus().getValue();
  if (congMod.ule(APInt::getZero(congMod.getBitWidth()))) {
    return emitOpError() << "congruence modulus must be positive";
  }

  if (outRingCmod.ule(congMod.zextOrTrunc(outRingCmod.getBitWidth()))) {
    return emitOpError()
           << "output ring cmod must be larger than congruence modulus";
  }

  return success();
}

}  // namespace mlir::heir::poly_ext
