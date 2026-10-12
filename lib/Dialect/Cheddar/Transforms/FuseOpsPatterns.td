#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPSPATTERNS_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPSPATTERNS_TD_

include "lib/Dialect/Cheddar/IR/CheddarOps.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">, "values are equal">;

def HasOneUse : Constraint<
  CPred<"$0.size() == 1 && (*$0.begin()).hasOneUse()">,
  "has a single result with one use"
>;

def NotFollowedBySingleUseRescale : Constraint<
  CPred<"!isFollowedBySingleUseRescale($0)">,
  "not followed by a single-use rescale in the same context"
>;

def HasConstantDistance : Constraint<
  CPred<"static_cast<bool>(getConstantDistance($_builder, $0, $1))">,
  "has constant rotation distance"
>;

def GetConstantDistance : NativeCodeCall<
  "getConstantDistance($_builder, $0, $1)"
>;

def FuseMultRelinRescale : Pat<
  (Cheddar_RescaleOp
    $ctx,
    (Cheddar_RelinearizeOp:$relin
      $relinCtx,
      (Cheddar_MultOp:$mult $multCtx, $lhs, $rhs, $multOut),
      $key,
      $relinOut),
    $rescaleOut),
  (Cheddar_HMultOp
    $ctx, $lhs, $rhs, $key, $rescaleOut, ConstantAttr<BoolAttr, "true">),
  [(HasOneUse $relin), (HasOneUse $mult),
   (Equal $ctx, $relinCtx), (Equal $ctx, $multCtx)],
  [],
  (addBenefit 3)
>;

def FuseMultRelinRescaleFused : Pat<
  (Cheddar_RelinearizeRescaleOp
    $ctx,
    (Cheddar_MultOp:$mult $multCtx, $lhs, $rhs, $multOut),
    $key,
    $out),
  (Cheddar_HMultOp
    $ctx, $lhs, $rhs, $key, $out, ConstantAttr<BoolAttr, "true">),
  [(HasOneUse $mult), (Equal $ctx, $multCtx)],
  [],
  (addBenefit 2)
>;

def FuseMultRelin : Pat<
  (Cheddar_RelinearizeOp:$relin
    $ctx,
    (Cheddar_MultOp:$mult $multCtx, $lhs, $rhs, $multOut),
    $key,
    $relinOut),
  (Cheddar_HMultOp
    $ctx, $lhs, $rhs, $key, $relinOut, ConstantAttr<BoolAttr, "false">),
  [(HasOneUse $mult), (Equal $ctx, $multCtx),
   (NotFollowedBySingleUseRescale $relin)]
>;

def FuseHRotAddLHS : Pat<
  (Cheddar_AddOp
    $ctx,
    (Cheddar_HRotOp:$rot $rotCtx, $ui, $input, $rotOut, $dynDist, $staticDist),
    $other,
    $addOut),
  (Cheddar_HRotAddOp
    $ctx, $ui, $input, $other, $addOut,
    (GetConstantDistance $dynDist, $staticDist)),
  [(HasOneUse $rot), (Equal $ctx, $rotCtx),
   (HasConstantDistance $dynDist, $staticDist)]
>;

def FuseHRotAddRHS : Pat<
  (Cheddar_AddOp
    $ctx,
    $other,
    (Cheddar_HRotOp:$rot $rotCtx, $ui, $input, $rotOut, $dynDist, $staticDist),
    $addOut),
  (Cheddar_HRotAddOp
    $ctx, $ui, $input, $other, $addOut,
    (GetConstantDistance $dynDist, $staticDist)),
  [(HasOneUse $rot), (Equal $ctx, $rotCtx),
   (HasConstantDistance $dynDist, $staticDist)]
>;

def FuseHConjAddLHS : Pat<
  (Cheddar_AddOp
    $ctx,
    (Cheddar_HConjOp:$conj $conjCtx, $ui, $input, $conjOut),
    $other,
    $addOut),
  (Cheddar_HConjAddOp $ctx, $ui, $input, $other, $addOut),
  [(HasOneUse $conj), (Equal $ctx, $conjCtx)]
>;

def FuseHConjAddRHS : Pat<
  (Cheddar_AddOp
    $ctx,
    $other,
    (Cheddar_HConjOp:$conj $conjCtx, $ui, $input, $conjOut),
    $addOut),
  (Cheddar_HConjAddOp $ctx, $ui, $input, $other, $addOut),
  [(HasOneUse $conj), (Equal $ctx, $conjCtx)]
>;

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPSPATTERNS_TD_
