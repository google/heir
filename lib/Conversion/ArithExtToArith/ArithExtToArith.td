#ifndef LIB_CONVERSION_ARITHEXTTOARITH_ARITHEXTTOARITH_TD_
#define LIB_CONVERSION_ARITHEXTTOARITH_ARITHEXTTOARITH_TD_

include "lib/DRR/Utils.td"
include "lib/Dialect/ArithExt/IR/ArithExtOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ArithExtToArith : Pass<"arith-ext-to-arith", "ModuleOp"> {
  let summary = "Lower `arith_ext` to standard `arith`.";

  let description = [{
    This pass lowers the `arith_ext` dialect to their `arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::arith_ext::ArithExtDialect",
  ];
}

// Using DRR to generate the lowering patterns for specific operations

defvar DefGE = ConstantEnumCase<Arith_CmpIPredicateAttr, "uge">;

def ConvertSubIfGE : Pattern<
  (ArithExt_SubIfGEOp $x, $y),
  [
    (Arith_SubIOp:$subOp $x, $y, DefOverflow),
    (Arith_CmpIOp:$cmpOp DefGE, $x, $y),
    (SelectOp $cmpOp, $subOp, $x)
  ]
>;


def HasEnoughSpaceAddSub: Constraint<CPred<"llvm::cast<IntegerType>(getElementTypeOrSelf($_self.getType())).getWidth() >= ($0.getValue() - 1).getActiveBits() + 1">,
"underlying type is sufficient for modular add/sub operation without overflow">;

def HasEnoughSpaceMul: Constraint<CPred<"llvm::cast<IntegerType>(getElementTypeOrSelf($_self.getType())).getWidth() >= 2 * ($0.getValue() - 1).getActiveBits()">,
 "underlying type is sufficient for modular mul operation without overflow">;

def CastModulusAttributeAddSub : NativeCodeCall<"modulusHelper($0,$1,false)">;
def CastModulusAttributeMul : NativeCodeCall<"modulusHelper($0,$1,true)">;

def ConvertAddSimple : Pattern<
  (ArithExt_AddOp:$op $x, $y, $mod),
  [
    (Arith_AddIOp:$add $x, $y, DefOverflow),
    (Arith_RemUIOp $add, (Arith_ConstantOp (CastModulusAttributeAddSub $mod, $x)))
  ],
  [(HasEnoughSpaceAddSub:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertSubSimple : Pattern<
  (ArithExt_SubOp:$op $x, $y, $mod),
  [
    (Arith_SubIOp:$sub $x, $y, DefOverflow),
    (Arith_RemUIOp $sub, (Arith_ConstantOp (CastModulusAttributeAddSub $mod, $x)))
  ],
  [(HasEnoughSpaceAddSub:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertMulSimple : Pattern<
  (ArithExt_MulOp:$op $x, $y, $mod),
  [
    (Arith_MulIOp:$mul $x, $y, DefOverflow),
    (Arith_RemUIOp $mul, (Arith_ConstantOp (CastModulusAttributeMul $mod, $x)))
  ],
  [(HasEnoughSpaceMul:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertMacSimple : Pattern<
  (ArithExt_MacOp:$op $x, $y, $acc, $mod),
  [
    (Arith_MulIOp:$mul $x, $y, DefOverflow),
    (Arith_AddIOp:$add $mul, $acc, DefOverflow),
    (Arith_RemUIOp $add, (Arith_ConstantOp (CastModulusAttributeMul $mod, $x)))
  ],
  [(HasEnoughSpaceMul:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertAdd : Pattern<
  (ArithExt_AddOp $x, $y, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeAddSub $mod, $x)),
    (Arith_AddIOp:$add
      (Arith_ExtUIOp $x,
        (returnType $newmod)),
      (Arith_ExtUIOp $y,
        (returnType $newmod)),
      DefOverflow),
    (Arith_TruncIOp:$res
      (Arith_RemUIOp $add, $newmod))
  ]
>;

def ConvertSub : Pattern<
  (ArithExt_SubOp $x, $y, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeAddSub $mod, $x)),
    (Arith_SubIOp:$sub
      (Arith_ExtUIOp $x,
        (returnType $newmod)),
      (Arith_ExtUIOp $y,
        (returnType $newmod)),
      DefOverflow),
    (Arith_TruncIOp:$res
      (Arith_RemUIOp $sub, $newmod))
  ]
>;

def ConvertMul : Pattern<
  (ArithExt_MulOp $x, $y, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeMul $mod, $x)),
    (Arith_MulIOp:$mul
      (Arith_ExtUIOp $x,
        (returnType $newmod)),
      (Arith_ExtUIOp $y,
        (returnType $newmod)),
      DefOverflow),
    (Arith_TruncIOp:$res
      (Arith_RemUIOp $mul, $newmod))
  ]
>;

def ConvertMac : Pattern<
  (ArithExt_MacOp $x, $y, $acc, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeMul $mod, $x)),
    (Arith_MulIOp:$mul
      (Arith_ExtUIOp $x,
        (returnType $newmod)),
      (Arith_ExtUIOp $y,
        (returnType $newmod)),
      DefOverflow),
    (Arith_AddIOp:$add $mul,
      (Arith_ExtUIOp:$extacc $acc, (returnType $newmod)), DefOverflow),
    (Arith_TruncIOp:$res
      (Arith_RemUIOp $add, $newmod))
  ]
>;

#endif  // LIB_CONVERSION_ARITHEXTTOARITH_ARITHEXTTOARITH_TD_
