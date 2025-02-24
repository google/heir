#ifndef LIB_DIALECT_MGMT_IR_MGMTCANONICALIZATION_TD_
#define LIB_DIALECT_MGMT_IR_MGMTCANONICALIZATION_TD_

include "MgmtOps.td"
include "lib/Utils/DRR/Utils.td"
include "mlir/IR/PatternBase.td"

// mod reduce should be put after adjust scale
def ModReduceAfterAdjustScale : Pat<
  (Mgmt_AdjustScaleOp:$adjustScaleOp (Mgmt_ModReduceOp:$modReduceOp $input), $scale),
  (Mgmt_ModReduceOp (Mgmt_AdjustScaleOp $input, $scale)),
  [(HasOneUse:$adjustScaleOp), (HasOneUse:$modReduceOp)]
>;

// level reduce should be put after mod reduce
def ModReduceAfterLevelReduce : Pat<
  (Mgmt_LevelReduceOp:$levelReduceOp (Mgmt_ModReduceOp:$modReduceOp $input), $levelToDrop),
  (Mgmt_ModReduceOp (Mgmt_LevelReduceOp $input, $levelToDrop)),
  [(HasOneUse:$levelReduceOp), (HasOneUse:$modReduceOp)]
>;

// adjust scale should be put after level reduce
def AdjustScaleAfterLevelReduce : Pat<
  (Mgmt_LevelReduceOp:$levelReduceOp (Mgmt_AdjustScaleOp:$adjustScaleOp $input, $scale), $levelToDrop),
  (Mgmt_AdjustScaleOp (Mgmt_LevelReduceOp $input, $levelToDrop), $scale),
  [(HasOneUse:$levelReduceOp), (HasOneUse:$adjustScaleOp)]
>;

def AddI64Attr : NativeCodeCall<"$_builder.getI64IntegerAttr($0.getValue().getLimitedValue() + $1.getValue().getLimitedValue())">;

def MergeLevelReduce : Pat<
  (Mgmt_LevelReduceOp:$levelReduceOp2 (Mgmt_LevelReduceOp:$levelReduceOp1 $input, $levelToDrop1), $levelToDrop2),
  (Mgmt_LevelReduceOp $input, (AddI64Attr $levelToDrop1, $levelToDrop2)),
  [(HasOneUse:$levelReduceOp1), (HasOneUse:$levelReduceOp2)]
>;

def LevelOne : NativeCodeCall<"$_builder.getI64IntegerAttr(1)">;

// Technically this transformation is not equivalent, as two mod reduce ops will
// change the scaling factor twice. This assumes there is an AdjustScaleOp somewhere
// in the line to correct the scaling factor.
def MergeModReduce : Pat<
  (Mgmt_ModReduceOp:$modReduceOp2 (Mgmt_ModReduceOp:$modReduceOp1 $input)),
  (Mgmt_ModReduceOp (Mgmt_LevelReduceOp $input, (LevelOne))),
  [(HasOneUse:$modReduceOp1), (HasOneUse:$modReduceOp2)]
>;

#endif  // LIB_DIALECT_MGMT_IR_MGMTCANONICALIZATION_TD_
