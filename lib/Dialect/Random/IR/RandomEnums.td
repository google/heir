#ifndef LIB_DIALECT_RANDOM_IR_RANDOMENUM_TD_
#define LIB_DIALECT_RANDOM_IR_RANDOMENUM_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "lib/Dialect/Random/IR/RandomDialect.td"

def DistributionEnumAttr : I64EnumAttr<"Distribution", "An enum attribute representing a random distribution", [
  I64EnumAttrCase<"uniform", 0>,
  I64EnumAttrCase<"gaussian", 1>
]> {
    let cppNamespace = "::mlir::heir::random";
}

#endif  // LIB_DIALECT_RANDOM_IR_RANDOMENUM_TD_
