#ifndef LIB_DIALECT_CGGI_TRANSFORMS_DECOMPOSEOPERATIONS_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_DECOMPOSEOPERATIONS_TD_

include "lib/Dialect/CGGI/IR/CGGIOps.td"
include "lib/Dialect/LWE/IR/LWEOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/CommonAttrConstraints.td"

defvar kBinaryGateLutWidth = 4;
defvar kAndLut = 8;
defvar kOrLut = 14;
defvar kXorLut = 6;

class GetIntegerTyForLWEEncoding<int i> : NativeCodeCall<
      "$_builder.getIntegerAttr($_builder.getIntegerType("
          "cast<lwe::LWECiphertextType>($0.getType()).getPlaintextSpace()"
          ".getRing().getCoefficientType().getIntOrFloatBitWidth()),"
          # i # ")">;

class GetIndexAttr<int i> : NativeCodeCall<
      "$_builder.getIndexAttr(" # i # ")">;

class GetLookupTableAttr<int i, int width> : NativeCodeCall<
      "$_builder.getIntegerAttr($_builder.getIntegerType("
          # width # ", /*isSigned=*/false), " # i # ")">;

def ExpandAndOp : Pattern<
  (CGGI_AndOp:$andOp $lhs, $rhs),
  [
    // Transform into linear LWE operations and a programmable bootstrap
    (CGGI_ScalarShiftLeftOp:$mul_rhs $rhs, (GetIndexAttr<1>)),
    (LWE_AddOp:$add $mul_rhs, $lhs),
    (CGGI_ProgrammableBootstrapOp:$res $add, (GetLookupTableAttr<kAndLut, kBinaryGateLutWidth>))
  ]
>;

def ExpandOrOp : Pattern<
  (CGGI_OrOp:$orOp $lhs, $rhs),
  [
    // Transform into linear LWE operations and a programmable bootstrap
    (CGGI_ScalarShiftLeftOp:$mul_rhs $rhs, (GetIndexAttr<1>)),
    (LWE_AddOp:$add $mul_rhs, $lhs),
    (CGGI_ProgrammableBootstrapOp:$res $add, (GetLookupTableAttr<kOrLut, kBinaryGateLutWidth>))
  ]
>;

def ExpandXOrOp : Pattern<
  (CGGI_XorOp:$orOp $lhs, $rhs),
  [
    // Transform into linear LWE operations and a programmable bootstrap
    (CGGI_ScalarShiftLeftOp:$mul_rhs $rhs, (GetIndexAttr<1>)),
    (LWE_AddOp:$add $mul_rhs, $lhs),
    (CGGI_ProgrammableBootstrapOp:$res $add, (GetLookupTableAttr<kXorLut, kBinaryGateLutWidth>))
  ]
>;

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_DECOMPOSEOPERATIONS_TD_
