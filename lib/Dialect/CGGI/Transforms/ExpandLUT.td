#ifndef LIB_DIALECT_CGGI_TRANSFORMS_EXPANDLUT_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_EXPANDLUT_TD_

include "lib/Dialect/CGGI/IR/CGGIOps.td"
include "lib/Dialect/LWE/IR/LWEOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/CommonAttrConstraints.td"

class GetIntegerTyForLWEEncoding<int i> : NativeCodeCall<
      "$_builder.getIntegerAttr($_builder.getIntegerType("
          "widthFromEncodingAttr("
              "cast<lwe::LWECiphertextType>($0.getType()).getEncoding())),"
          # i # ")">;

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

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_EXPANDLUT_TD_
