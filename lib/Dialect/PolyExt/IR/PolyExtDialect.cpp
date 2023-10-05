#include "include/Dialect/PolyExt/IR/PolyExtDialect.h"

#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "include/Dialect/Poly/IR/Polynomial.h"
#include "include/Dialect/PolyExt/IR/PolyExtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions.
#include "include/Dialect/PolyExt/IR/PolyExtDialect.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/PolyExt/IR/PolyExtOps.cpp.inc"

namespace mlir::heir::poly_ext {

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void PolyExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/PolyExt/IR/PolyExtOps.cpp.inc"
      >();
}

poly::PolyType getPolyType(Type t) {
  if (auto tTnsr = t.dyn_cast<RankedTensorType>()) {
    return tTnsr.getElementType().cast<poly::PolyType>();
  }
  return t.cast<poly::PolyType>();
}

LogicalResult CModSwitchOp::verify() {
  auto xRing = getPolyType(getX().getType()).getRing();
  auto outRing = getPolyType(getOutput().getType()).getRing();

  if (xRing.getIdeal() != outRing.getIdeal()) {
    return emitOpError("input and output rings ideals must be the same");
  }

  if (xRing.getCmod().ule(outRing.getCmod())) {
    return emitOpError("input ring cmod must be larger than output ring cmod");
  }

  if (getCongruenceModulus().ule(0)) {
    return emitOpError("congruence modulus must be positive");
  }

  if (outRing.getCmod().ule(getCongruenceModulus())) {
    return emitOpError(
        "output ring cmod must be larger than congruence modulus");
  }

  return success();
}

}  // namespace mlir::heir::poly_ext
