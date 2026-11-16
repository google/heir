#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_CONSTRAINTS_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_CONSTRAINTS_TD_

include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Buffer Type Constraints
//===----------------------------------------------------------------------===//

// Statically shaped MemRef of a given rank.
class CheddarGPU_StaticMemRefOfRank<list<Type> elems, int rank, string descr = "",
                                    list<Pred> extra = []>
    : ConfinedType<MemRefRankOf<elems, [rank]>, !listconcat([HasStaticShapePred], extra),
                   !if(!empty(descr), descr, "statically shaped " # MemRefRankOf<elems, [rank]>.summary),
                   "::mlir::MemRefType">;

// 2D statically shaped polynomial buffer of i32 or i64 elements.
def CheddarGPU_PolyBuffer : CheddarGPU_StaticMemRefOfRank<[I32, I64], 2,
    "statically shaped 2D memref<L x N x iW> polynomial buffer (W = 32 or 64)">;

// 1D statically shaped per-limb constant vector of i32 or i64 elements.
def CheddarGPU_LimbVector : CheddarGPU_StaticMemRefOfRank<[I32, I64], 1,
    "statically shaped 1D memref<L x iW> per-limb constant vector (W = 32 or 64)">;

//===----------------------------------------------------------------------===//
// Dimension Helpers and PredOpTraits
//===----------------------------------------------------------------------===//

class CheddarGPU_Dim<string v, int d>
    : StrFunc<"::llvm::cast<::mlir::MemRefType>($" # v # ".getType()).getDimSize(" # d # ")">;

class CheddarGPU_SameDim<list<string> names, int d, string what>
    : AllMatchSameOperatorTrait<names,
        "::llvm::cast<::mlir::MemRefType>($_self.getType()).getDimSize(" # d # ")", what>;

class CheddarGPU_SameDegree<list<string> n>
    : CheddarGPU_SameDim<n, 1, "polynomial degree (dim 1)">;

class CheddarGPU_DimRel<string lhs, int ld, string cmp, string rhs, int rd, string descr>
    : PredOpTrait<descr, CPred<CheddarGPU_Dim<lhs, ld>.result # " " # cmp # " "
                               # CheddarGPU_Dim<rhs, rd>.result>>;

class CheddarGPU_PerLimbOf<string vec, string poly>
    : CheddarGPU_DimRel<vec, 0, "==", poly, 0,
        "'" # vec # "' has one entry per limb of '" # poly # "'">;

//===----------------------------------------------------------------------===//
// Form Constraints
//===----------------------------------------------------------------------===//

class CheddarGPU_FormsMatch<list<string> a>
    : AllMatchSameOperatorTrait<a, "$_self", "form">;

//===----------------------------------------------------------------------===//
// Attribute Constraints
//===----------------------------------------------------------------------===//

def CheddarGPU_OffsetCount : AttrConstraint<CPred<
    "::llvm::cast<::mlir::DenseI64ArrayAttr>($_self).asArrayRef()[0] >= 0 && "
    "::llvm::cast<::mlir::DenseI64ArrayAttr>($_self).asArrayRef()[1] > 0">,
    "of the form [offset >= 0, count > 0]">;

def CheddarGPU_LimbRangeAttr
    : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>, CheddarGPU_OffsetCount]>;

def CheddarGPU_NonNegI64Attr : ConfinedAttr<I64Attr, [IntNonNegative]>;

//===----------------------------------------------------------------------===//
// Native Traits
//===----------------------------------------------------------------------===//

def CheddarGPU_LimbRangesWithin : NativeOpTrait<"LimbRangesWithin"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

def CheddarGPU_LimbRangesWithinDest : NativeOpTrait<"LimbRangesWithin"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

def CheddarGPU_InOutLimbRangesWithin : NativeOpTrait<"InOutLimbRangesWithin"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

def CheddarGPU_InOutCountsMatch : NativeOpTrait<"InOutCountsMatch"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_CONSTRAINTS_TD_
