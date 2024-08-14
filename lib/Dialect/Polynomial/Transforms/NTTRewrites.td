#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_TD_

include "lib/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Polynomial/IR/Polynomial.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"

def GetRingAttr : NativeCodeCall<
      "(dyn_cast<::mlir::polynomial::PolynomialType>($0.getType())).getRing()">;

def GetRingModAttr : NativeCodeCall<
      "(dyn_cast<::mlir::polynomial::PolynomialType>($0.getType())).getRing().getCoefficientModulus()">;

def InputTensorType : NativeCodeCall<
      "RankedTensorType::get({$0.getPolynomialModulus().getPolynomial().getDegree()},"
      " IntegerType::get($_builder.getContext(),"
      "  ($0.getCoefficientModulus().getValue() - 1).getActiveBits()),"
      " $0)">;

def CreateCModConstant : NativeCodeCall<
      "$_builder.create<arith::ConstantOp>($0.getLoc(), $2,"
      " DenseElementsAttr::get($2,"
      "  $1.getCoefficientModulus().getValue().sextOrTrunc($2.getElementTypeBitWidth())))">;

def HasDegreePowerOfTwo : Constraint<
    CPred<"APInt(64, (dyn_cast<::mlir::polynomial::PolynomialType>($0.getType())).getRing()"
          ".getPolynomialModulus().getPolynomial().getDegree()).isPowerOf2()">,
    "rings are NTT compatible">;

def Nullptr
  : NativeCodeCall<"nullptr">;

def NTTRewritePolyMul : Pattern<
  (Polynomial_MulOp:$mulOp $p1, $p2),
  [
    // Transform to NTT point-value representation
    (Polynomial_NTTOp:$p1NTT $p1, (Nullptr),
      (returnType (InputTensorType (GetRingAttr $p1)))),
    (Polynomial_NTTOp:$p2NTT $p2, (Nullptr),
      (returnType (InputTensorType (GetRingAttr $p2)))),

    // Compute elementwise multiplication modulo cmod
    (ModArith_MulOp:$mulNTT $p1NTT, $p2NTT, (GetRingModAttr $p1)),

    // Compute inverse transform back to coefficient representation
    (Polynomial_INTTOp:$res $mulNTT, (Nullptr))
  ],
  [
    (HasDegreePowerOfTwo $p1)
  ]
>;

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_TD_
