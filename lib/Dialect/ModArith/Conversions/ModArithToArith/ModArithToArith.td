#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
#define LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_

include "lib/Utils/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ModArithToArith : Pass<"mod-arith-to-arith", "ModuleOp"> {
  let summary = "Lower `mod_arith` to standard `arith`.";

  let description = [{
    This pass lowers the `mod_arith` dialect to their `arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

// Using DRR to generate the lowering patterns for specific operations

defvar DefGE = ConstantEnumCase<Arith_CmpIPredicateAttr, "uge">;

def ConvertSubIfGE : Pattern<
  (ModArith_SubIfGEOp $x, $y),
  [
    (Arith_SubIOp:$subOp $x, $y, DefOverflow),
    (Arith_CmpIOp:$cmpOp DefGE, $x, $y),
    (SelectOp $cmpOp, $subOp, $x)
  ]
>;


def HasEnoughSpaceAddSub: Constraint<CPred<"llvm::cast<IntegerType>(getElementTypeOrSelf($_self.getType())).getWidth() >= ($0.getValue() - 1).getActiveBits() + 1">,
"underlying type is sufficient for modular add/sub operation without overflow">;

def HasEnoughSpaceReduce: Constraint<CPred<"llvm::cast<IntegerType>(getElementTypeOrSelf($_self.getType())).getWidth() - 1 >= $0.getValue().getActiveBits()">,
"underlying type is sufficient for modular reduce operation without signedness issue">;

def HasEnoughSpaceMul: Constraint<CPred<"llvm::cast<IntegerType>(getElementTypeOrSelf($_self.getType())).getWidth() >= 2 * ($0.getValue() - 1).getActiveBits()">,
 "underlying type is sufficient for modular mul operation without overflow">;

def ModulusIsNatural : Constraint<CPred<
  "llvm::cast<IntegerType>(getElementTypeOrSelf($_self.getType())).getWidth() "
  "  == ($0.getValue() - 1).getActiveBits()"
  "  && $0.getValue().isPowerOf2()">, "modulus is the natural modulus: 2 ** bitwidth">;

def CastModulusAttributeAddSub : NativeCodeCall<"modulusHelper($0,$1,false)">;
def CastModulusAttributeMul : NativeCodeCall<"modulusHelper($0,$1,true)">;
def CastModulusAttributeReduce : NativeCodeCall<"modulusHelper($0,$1,false,true)">;

def ConvertAddNatural : Pattern<
  (ModArith_AddOp:$op $x, $y, $mod),
  [
    (Arith_AddIOp:$add $x, $y, DefOverflow),
  ],
  [(ModulusIsNatural:$op $mod)],
  [],
  (addBenefit 4)
>;

def ConvertSubNatural : Pattern<
  (ModArith_SubOp:$op $x, $y, $mod),
  [
    (Arith_SubIOp:$add $x, $y, DefOverflow),
  ],
  [(ModulusIsNatural:$op $mod)],
  [],
  (addBenefit 4)
>;

def ConvertMulNatural : Pattern<
  (ModArith_MulOp:$op $x, $y, $mod),
  [
    (Arith_MulIOp:$add $x, $y, DefOverflow),
  ],
  [(ModulusIsNatural:$op $mod)],
  [],
  (addBenefit 4)
>;

def ConvertReduceNatural : Pattern<
  (ModArith_ReduceOp:$op $x, $mod),
  [
    (replaceWithValue $x),
  ],
  [(ModulusIsNatural:$op $mod)],
  [],
  (addBenefit 4)
>;

def ConvertAddSimple : Pattern<
  (ModArith_AddOp:$op $x, $y, $mod),
  [
    (Arith_AddIOp:$add $x, $y, DefOverflow),
    (Arith_RemUIOp $add, (Arith_ConstantOp (CastModulusAttributeAddSub $mod, $x)))
  ],
  [(HasEnoughSpaceAddSub:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertSubSimple : Pattern<
  (ModArith_SubOp:$op $x, $y, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeAddSub $mod, $x)),
    (Arith_SubIOp:$sub $x, $y, DefOverflow),
    (Arith_AddIOp:$shift $sub, $newmod, DefOverflow),
    (Arith_RemUIOp $shift, $newmod)
  ],
  [(HasEnoughSpaceAddSub:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertMulSimple : Pattern<
  (ModArith_MulOp:$op $x, $y, $mod),
  [
    (Arith_MulIOp:$mul $x, $y, DefOverflow),
    (Arith_RemUIOp $mul, (Arith_ConstantOp (CastModulusAttributeMul $mod, $x)))
  ],
  [(HasEnoughSpaceMul:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertMacSimple : Pattern<
  (ModArith_MacOp:$op $x, $y, $acc, $mod),
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
  (ModArith_AddOp $x, $y, $mod),
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
  (ModArith_SubOp $x, $y, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeAddSub $mod, $x)),
    (Arith_SubIOp:$sub
      (Arith_ExtUIOp $x,
        (returnType $newmod)),
      (Arith_ExtUIOp $y,
        (returnType $newmod)),
      DefOverflow),
    (Arith_AddIOp:$shift $sub, $newmod, DefOverflow),
    (Arith_TruncIOp:$res
      (Arith_RemUIOp $shift, $newmod))
  ]
>;

def ConvertMul : Pattern<
  (ModArith_MulOp $x, $y, $mod),
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
  (ModArith_MacOp $x, $y, $acc, $mod),
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

def ConvertReduceSimple : Pattern<
  (ModArith_ReduceOp:$op $x, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeReduce $mod, $x)),
    (Arith_RemUIOp (Arith_AddIOp (Arith_RemSIOp $x, $newmod), $newmod, DefOverflow), $newmod)
  ],
  [(HasEnoughSpaceReduce:$op $mod)],
  [],
  (addBenefit 2)
>;

def ConvertReduce : Pattern<
  (ModArith_ReduceOp $x, $mod),
  [
    (Arith_ConstantOp:$newmod (CastModulusAttributeReduce $mod, $x)),
    (Arith_RemSIOp:$rems
      (Arith_ExtSIOp $x,
        (returnType $newmod)),
      $newmod),
    (Arith_AddIOp:$remu $rems, $newmod, DefOverflow),
    (Arith_TruncIOp:$res
      (Arith_RemUIOp $remu, $newmod))
  ]
>;

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
