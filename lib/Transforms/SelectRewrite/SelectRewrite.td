#ifndef LIB_TRANSFORMS_SELECTREWRITE_SELECTREWRITE_TD_
#define LIB_TRANSFORMS_SELECTREWRITE_SELECTREWRITE_TD_

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "lib/Utils/DRR/Utils.td"
include "mlir/Pass/PassBase.td"
include "mlir/IR/PatternBase.td"

def SelectRewrite : Pass<"select-rewrite"> {
  let summary = "Rewrites arith.select to a CMUX style expression";
  let description = [{
    "This pass rewrites arith.select %c, %t, %f to %c * %t + (1 - %c) * %f.
     It supports all three variants of arith.select: scalar, shaped, and mixed types.
     In the latter case, it will broadcast/splat the scalar condition value to the required shape."
  }];
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
  ];
}

// If the first input is a value of scalar type and the second is one of shaped type,
// return the first value broadcast/splat to the same shape as the second value,
// preserving the "element" type of the first value.
def matchShapedType : NativeCodeCall<"matchShapedType($_builder, $0, $1)">;

// Get either an IntegerAttr or DenseIntegerAtr of the same type as the input
def getMatchingOne : NativeCodeCall<"getMatchingOne($0)">;

// Rewrites arith.select %c, %t, %f to %c * %t + (1 - %c) * %f
def SelectRewritePatternInt : Pattern<
  (SelectOp $cond, SignlessIntegerLike:$true_value, $false_value),
  [
    (matchShapedType:$new_cond $cond, $true_value),
    (Arith_ConstantOp:$one (getMatchingOne $new_cond, $true_value)),
    (Arith_SubIOp:$not_cond $one, $new_cond, DefOverflow),
    (Arith_ExtUIOp:$ext_cond $new_cond, (returnType $true_value)),
    (Arith_MulIOp:$true_mul $ext_cond, $true_value, DefOverflow),
    (Arith_ExtUIOp:$ext_not_cond $not_cond, (returnType $false_value)),
    (Arith_MulIOp:$false_mul $ext_not_cond, $false_value, DefOverflow),
    (Arith_AddIOp $true_mul, $false_mul, DefOverflow)
  ]
>;

// Rewrites arith.select %c, %t, %f to %c * %t + (1 - %c) * %f
def SelectRewritePatternFloat : Pattern<
  (SelectOp $cond, FloatLike:$true_value, $false_value),
  [
    (Arith_ConstantOp:$one (getMatchingOne $cond)),
    (Arith_SubIOp:$not_cond $one, $cond, DefOverflow),
    (Arith_UIToFPOp:$ext_cond $cond, (returnType $true_value)),
    (Arith_MulFOp:$true_mul $ext_cond, $true_value, DefFastMath),
    (Arith_UIToFPOp:$ext_not_cond $not_cond, (returnType $false_value)),
    (Arith_MulFOp:$false_mul $ext_not_cond, $false_value, DefFastMath),
    (Arith_AddFOp $true_mul, $false_mul, DefFastMath)
  ]
>;


#endif  // LIB_TRANSFORMS_SELECTREWRITE_SELECTREWRITE_TD_
