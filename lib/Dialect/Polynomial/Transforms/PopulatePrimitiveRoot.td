#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POPULATEPRIMITIVEROOT_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POPULATEPRIMITIVEROOT_TD_

include "lib/Utils/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Polynomial/IR/Polynomial.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"

def GetRingAttr : NativeCodeCall<
      "(dyn_cast<::mlir::polynomial::PolynomialType>($0.getType())).getRing()">;

def PopulatePrimitiveRoot
  : NativeCodeCall<"populatePrimitiveRoot($0)">;

def Nullptr
  : NativeCodeCall<"nullptr">;

def RootUnspecified
  : Constraint<CPred<"$0 == nullptr">>;

def RootAvailable
  : Constraint<CPred<"populatePrimitiveRoot($0) != nullptr">>;

def PopulatePrimitiveRootNTT : Pattern<
  (Polynomial_NTTOp $p, $r),
  [
    (Polynomial_NTTOp $p, (PopulatePrimitiveRoot $p)),
  ],
  [
    (RootUnspecified $r),
    (RootAvailable $p)
  ]
>;

def PopulatePrimitiveRootINTT : Pattern<
  (Polynomial_INTTOp:$p $t, $r),
  [
    (Polynomial_INTTOp $t, (PopulatePrimitiveRoot $p)),
  ],
  [
    (RootUnspecified $r),
    (RootAvailable $p)
  ]
>;

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_POPULATEPRIMITIVEROOT_TD_
