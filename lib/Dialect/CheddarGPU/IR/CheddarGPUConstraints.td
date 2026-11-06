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

def CheddarGPU_LimbRangesWithinDest : NativeOpTrait<"LimbRangesWithinDest"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_CONSTRAINTS_TD_
