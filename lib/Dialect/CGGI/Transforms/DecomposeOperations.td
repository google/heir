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
          "widthFromEncodingAttr("
              "cast<lwe::LWECiphertextType>($0.getType()).getEncoding())),"
          # i # ")">;

class GetIndexAttr<int i> : NativeCodeCall<
      "$_builder.getIndexAttr(" # i # ")">;

class GetLookupTableAttr<int i, int width> : NativeCodeCall<
      "$_builder.getIntegerAttr($_builder.getIntegerType("
          # width # ", /*isSigned=*/false), " # i # ")">;

def ExpandLUT2 : Pattern<
  (CGGI_Lut2Op:$lutOp $b, $a, $lookup_table),
  [
    // Transform into linear LWE operations and a programmable bootstrap
    (Arith_ConstantOp:$const2 (GetIntegerTyForLWEEncoding<2> $b)),
    (LWE_MulScalarOp:$mul_b $b, $const2),
    (LWE_AddOp:$add_ba $mul_b, $a),
    (CGGI_ProgrammableBootstrapOp:$res $add_ba, $lookup_table)
  ]
>;

def ExpandLUT3 : Pattern<
  (CGGI_Lut3Op:$lutOp $c, $b, $a, $lookup_table),
  [
    // Transform into linear LWE operations and a programmable bootstrap
    (Arith_ConstantOp:$const4 (GetIntegerTyForLWEEncoding<4> $c)),
    (Arith_ConstantOp:$const2 (GetIntegerTyForLWEEncoding<2> $b)),
    (LWE_MulScalarOp:$mul_c $c, $const4),
    (LWE_MulScalarOp:$mul_b $b, $const2),
    (LWE_AddOp:$add_cb $mul_c, $mul_b),
    (LWE_AddOp:$add_cba $add_cb, $a),
    (CGGI_ProgrammableBootstrapOp:$res $add_cba, $lookup_table)
  ]
>;

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
