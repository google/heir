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

// 2D or 1D statically shaped polynomial buffer or per-limb constant vector.
def CheddarGPU_PolyOrLimbBuffer : AnyTypeOf<
    [CheddarGPU_PolyBuffer, CheddarGPU_LimbVector],
    "statically shaped polynomial buffer or per-limb constant vector", "::mlir::MemRefType">;

// 2D statically shaped basis-conversion table of i32 or i64 elements.
def CheddarGPU_BConvTable : CheddarGPU_StaticMemRefOfRank<[I32, I64], 2,
    "statically shaped 2D memref<D x S x iW> basis-conversion table (W = 32 or 64)">;

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

// Active limb count of the destination: out_limb_range[1] if set, else dim 0.
class CheddarGPU_OutCount<string out> {
  string result = "(getOutLimbRange() ? (*getOutLimbRange())[1] : "
                  # CheddarGPU_Dim<out, 0>.result # ")";
}

// Active limb count of the source: in_limb_range[1] if set, else dim 0.
class CheddarGPU_InCount<string src> {
  string result = "(getInLimbRange() ? (*getInLimbRange())[1] : "
                  # CheddarGPU_Dim<src, 0>.result # ")";
}

class CheddarGPU_DimEquals<string v, int d, string expr, string descr>
    : PredOpTrait<descr, CPred<CheddarGPU_Dim<v, d>.result # " == (" # expr # ")">>;

class CheddarGPU_EachCoversOutCount<string v, string out, int onlyRank = 0> : PredOpTrait<
    "every " # !if(onlyRank, onlyRank # "D ", "") # "'" # v
        # "' has at least as many limbs as the active range of '" # out # "'",
    CPred<"::llvm::all_of($" # v # ".getTypes(), [&](::mlir::Type t) { auto m = "
          "::llvm::cast<::mlir::MemRefType>(t); return "
          # !if(onlyRank, "m.getRank() != " # onlyRank # " || ", "")
          # "m.getDimSize(0) >= " # CheddarGPU_OutCount<out>.result # "; })">>;

//===----------------------------------------------------------------------===//
// Variadic and Optional Operand Groups
//===----------------------------------------------------------------------===//

class CheddarGPU_SameCount<list<string> n>
    : AllMatchSameOperatorTrait<n, "$_self.size()", "count">;

class CheddarGPU_NonEmpty<string v>
    : PredOpTrait<"'" # v # "' is non-empty", CPred<"!$" # v # ".empty()">>;

class CheddarGPU_EachTypeIs<string v, string ref> : PredOpTrait<
    "every '" # v # "' has the type of '" # ref # "'",
    CPred<"::llvm::all_of($" # v # ".getTypes(), [&](::mlir::Type t) { return t == $"
          # ref # ".getType(); })">>;

class CheddarGPU_Each2DTypeIs<string v, string ref> : PredOpTrait<
    "every 2D '" # v # "' has the type of '" # ref # "'",
    CPred<"::llvm::all_of($" # v # ".getTypes(), [&](::mlir::Type t) { return "
          "::llvm::cast<::mlir::MemRefType>(t).getRank() != 2 || t == $" # ref # ".getType(); })">>;

class CheddarGPU_EachSameRingAs<string v, string ref> : PredOpTrait<
    "every '" # v # "' has the degree and word type of '" # ref # "'",
    CPred<"::llvm::all_of($" # v # ".getTypes(), [&](::mlir::Type t) { auto m = "
          "::llvm::cast<::mlir::MemRefType>(t); auto r = ::llvm::cast<::mlir::MemRefType>($"
          # ref # ".getType()); return m.getDimSize(1) == r.getDimSize(1) && "
          "m.getElementType() == r.getElementType(); })">>;

class CheddarGPU_Getter<string n> { string call = "get" # snakeCaseToCamelCase<n>.ret # "()"; }

class CheddarGPU_BothOrNeither<string a, string b> : PredOpTrait<
    "'" # a # "' and '" # b # "' are both present or both absent",
    CPred<"static_cast<bool>(" # CheddarGPU_Getter<a>.call # ") == static_cast<bool>("
          # CheddarGPU_Getter<b>.call # ")">>;

class CheddarGPU_LevelOrSentinel<string sentinelCond, string sentinelDescr> : PredOpTrait<
    "level is non-negative, or -1 when " # sentinelDescr,
    CPred<"static_cast<int64_t>($level) >= 0 || (static_cast<int64_t>($level) == -1 && ("
          # sentinelCond # "))">>;

class CheddarGPU_Requires<string a, string bCond, string descr>
    : PredOpTrait<descr, CPred<"!" # CheddarGPU_Getter<a>.call # " || (" # bCond # ")">>;

class CheddarGPU_OptionalSameRingAs<string opt, string ref> : PredOpTrait<
    "'" # opt # "' (if present) has the degree and word type of '" # ref # "'",
    CPred<"!" # CheddarGPU_Getter<opt>.call # " || (" # CheddarGPU_Dim<opt, 1>.result # " == "
          # CheddarGPU_Dim<ref, 1>.result # " && ::llvm::cast<::mlir::MemRefType>($" # opt
          # ".getType()).getElementType() == ::llvm::cast<::mlir::MemRefType>($" # ref # ".getType()).getElementType())">>;

class CheddarGPU_OptionalElemTypeMatches<string opt, string ref> : PredOpTrait<
    "'" # opt # "' (if present) has same element type as '" # ref # "'",
    CPred<"!" # CheddarGPU_Getter<opt>.call # " || ("
          # "::llvm::cast<::mlir::MemRefType>($" # opt # ".getType()).getElementType() == "
          # "::llvm::cast<::mlir::MemRefType>($" # ref # ".getType()).getElementType())">>;

class CheddarGPU_SpanEndWithinDim0<string span, string buf, bit optional = 1> : PredOpTrait<
    "end of '" # span # "' is within dim 0 of '" # buf # "'",
    CPred<!if(optional,
        "!" # CheddarGPU_Getter<span>.call # " || (*" # CheddarGPU_Getter<span>.call # ")[1] <= " # CheddarGPU_Dim<buf, 0>.result,
        CheddarGPU_Getter<span>.call # "[1] <= " # CheddarGPU_Dim<buf, 0>.result)>>;

class CheddarGPU_SpanLen<string attrName> {
  string result = "(" # CheddarGPU_Getter<attrName>.call # " ? ((*"
                  # CheddarGPU_Getter<attrName>.call # ")[1] - (*"
                  # CheddarGPU_Getter<attrName>.call # ")[0]) : 0)";
}

class CheddarGPU_SpanEndWithin<string attrName, string boundExpr, string descr>
    : PredOpTrait<descr,
        CPred<"!" # CheddarGPU_Getter<attrName>.call # " || (*"
              # CheddarGPU_Getter<attrName>.call # ")[1] <= (" # boundExpr # ")">>;

class CheddarGPU_OptionalArrayMatchesDim<string attrName, string v, int d>
    : PredOpTrait<
        "'" # attrName # "' (if present) length matches dim " # d # " of '" # v # "'",
        CPred<"!" # CheddarGPU_Getter<attrName>.call # " || static_cast<int64_t>("
              # CheddarGPU_Getter<attrName>.call # "->size()) == "
              # CheddarGPU_Dim<v, d>.result>>;

//===----------------------------------------------------------------------===//
// Form Constraints
//===----------------------------------------------------------------------===//

class CheddarGPU_FormsMatch<list<string> a>
    : AllMatchSameOperatorTrait<a, "$_self", "form">;

class CheddarGPU_ExactlyOneOf<string optAttr, string unitAttr> : PredOpTrait<
    "exactly one of '" # optAttr # "' and '" # unitAttr # "' is set",
    CPred<"$" # optAttr # ".has_value() != static_cast<bool>($" # unitAttr # ")">>;

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
def CheddarGPU_PosI64Attr : ConfinedAttr<I64Attr, [IntPositive]>;

def CheddarGPU_StartEnd : AttrConstraint<CPred<
    "::llvm::cast<::mlir::DenseI64ArrayAttr>($_self).asArrayRef()[0] >= 0 && "
    "::llvm::cast<::mlir::DenseI64ArrayAttr>($_self).asArrayRef()[1] >= "
    "::llvm::cast<::mlir::DenseI64ArrayAttr>($_self).asArrayRef()[0]">,
    "of the form [start >= 0, end >= start]">;

def CheddarGPU_SpanAttr
    : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>, CheddarGPU_StartEnd]>;

def CheddarGPU_PairAttr : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;

def CheddarGPU_InSplitAttr : ConfinedAttr<DenseI64ArrayAttr,
    [DenseArrayCount<2>, DenseArrayNonNegative<DenseI64ArrayAttr>]>;

//===----------------------------------------------------------------------===//
// Native Traits
//===----------------------------------------------------------------------===//

def CheddarGPU_NumAuxWithinOut : NativeOpTrait<"NumAuxWithinOut"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

// Verifies that each listed `[offset, count]` limb range fits within dim 0 of
// each listed operand (and within the module's prime chain), and that limb
// ranges are not applied to tensor operands.
//
// `checks` is a C++ braced-init-list of `{attrName, range, {operands...}}`
// entries, evaluated inside the concrete op class (so ODS accessors work).
class CheddarGPU_LimbRanges<code checks>
    : NativeOpTrait<"LimbRangesWithin", /*traits=*/[],
        /*extraOpDeclaration=*/[{
          ::llvm::SmallVector<::mlir::heir::cheddar_gpu::LimbRangeCheck>
          getLimbRangeChecks();
        }],
        /*extraOpDefinition=*/!strconcat(
          "::llvm::SmallVector<::mlir::heir::cheddar_gpu::LimbRangeCheck> ",
          "$cppClass::getLimbRangeChecks() { return ",
          checks,
          "; }")> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

def CheddarGPU_InOutCountsMatch : NativeOpTrait<"InOutCountsMatch"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

def CheddarGPU_NTTPrimeCounts : NativeOpTrait<"NTTPrimeCounts"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

def CheddarGPU_PadRangeWithinDest : NativeOpTrait<"PadRangeWithinDest"> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_CONSTRAINTS_TD_
