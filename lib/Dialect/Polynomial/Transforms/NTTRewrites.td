#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_TD_

include "lib/DRR/Utils.td"
include "lib/Dialect/Polynomial/IR/PolynomialOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"

def GetRingAttr : NativeCodeCall<
      "(dyn_cast<PolynomialType>($0.getType())).getRing()">;

def InputTensorType : NativeCodeCall<
      "RankedTensorType::get({$0.ideal().getDegree()},"
      " IntegerType::get($_builder.getContext(),"
      "  ($0.coefficientModulus() - 1).getActiveBits()),"
      " $0)">;

def IntermediateTensorType : NativeCodeCall<
      "RankedTensorType::get({$0.ideal().getDegree()},"
      " IntegerType::get($_builder.getContext(),"
      "  2 * ($0.coefficientModulus() - 1).getActiveBits()),"
      " $0)">;

def CreateCModConstant : NativeCodeCall<
      "$_builder.create<arith::ConstantOp>($0.getLoc(), $2,"
      " DenseElementsAttr::get($2,"
      "  $1.coefficientModulus().sextOrTrunc($2.getElementTypeBitWidth())))">;

// Computes (n & (n - 1)) == 0.
def HasDegreePowerOfTwo : Constraint<
    CPred<"APInt(64, (dyn_cast<PolynomialType>($0.getType())).getRing()"
          ".ideal().getDegree()).isPowerOf2()">,
    "rings are NTT compatible">;

def NTTRewritePolyMul : Pattern<
  (Polynomial_MulOp:$mulOp $p1, $p2),
  [
    // Transform to NTT point-value representation
    (Polynomial_NTTOp:$p1NTT $p1,
      (returnType (InputTensorType (GetRingAttr $p1)))),
    (Polynomial_NTTOp:$p2NTT $p2,
      (returnType (InputTensorType (GetRingAttr $p2)))),

    // Compute elementwise multiplication modulo cmod
    (Arith_MulIOp:$mulNTT
      (Arith_ExtUIOp $p1NTT,
        (returnType (IntermediateTensorType (GetRingAttr $p1)))),
      (Arith_ExtUIOp $p2NTT,
        (returnType (IntermediateTensorType (GetRingAttr $p2)))),
      DefOverflow),
    (Arith_TruncIOp:$resNTT
      (Arith_RemUIOp $mulNTT,
        (CreateCModConstant $mulNTT,
          (GetRingAttr $p1), (IntermediateTensorType (GetRingAttr $p2)))),
      (returnType (InputTensorType (GetRingAttr $p1)))),

    // Compute inverse transform back to coefficient representation
    (Polynomial_INTTOp:$res $resNTT)
  ],
  [
    (HasDegreePowerOfTwo $p1)
  ]
>;

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_NTTREWRITES_TD_
