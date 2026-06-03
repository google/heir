#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h"

#include <cmath>
#include <cstdio>
#include <string>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToEmitC/ArithToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/SCFToEmitC/SCFToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CHEDDARTOEMITC
#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

namespace {

using ::mlir::emitc::CallOpaqueOp;
using ::mlir::emitc::LValueType;
using ::mlir::emitc::MemberCallOpaqueOp;
using ::mlir::emitc::OpaqueAttr;
using ::mlir::emitc::OpaqueType;
using ::mlir::emitc::PointerType;
using ::mlir::emitc::VariableOp;
using ::mlir::emitc::VerbatimOp;

// Returns true if `t` is an opaque type whose textual name is one of the
// move-only CHEDDAR payload types. Used by the type converter (to decide
// memref-of-cheddar -> emitc.array vs memref-of-emitc.opaque), the
// DPS-lifting post-pass, and the load-elision pass.
//
// EvaluationKey/Plaintext/Ciphertext/Constant are the move-only CHEDDAR
// payload types that the out-param pattern produces (as `T out;` locals and,
// at function boundaries, as `const T&` inputs / `T&` out-params). EvkMap and
// Encoder are also non-copy-assignable at the C++ level, but they are never
// produced as out-params -- they only ever appear as inputs -- so they are
// handled separately by `isConstRefBoundaryOpaque` (const-ref arg tightening)
// rather than here.
bool isMoveOnlyOpaque(Type t, StringRef& nameOut) {
  auto opaqueT = dyn_cast<emitc::OpaqueType>(t);
  if (!opaqueT) return false;
  StringRef name = opaqueT.getValue();
  if (name == "Ciphertext<word>" || name == "Plaintext<word>" ||
      name == "Constant<word>" || name == "EvaluationKey<word>") {
    nameOut = name;
    return true;
  }
  return false;
}

// EvkMap is move-only (copy deleted, no move-assignment) and Encoder is a
// non-assignable view (it holds reference members). Both must be passed by
// `const T&` at a function boundary: a by-value parameter would force an
// unnecessary move/copy at the call site (and, for EvkMap, a copy is deleted).
// Unlike the payload types they are never returned via the out-param pattern,
// so this predicate is consulted only for input tightening.
bool isConstRefBoundaryOpaque(Type t, StringRef& nameOut) {
  auto opaqueT = dyn_cast<emitc::OpaqueType>(t);
  if (!opaqueT) return false;
  StringRef name = opaqueT.getValue();
  if (name == "EvkMap<word>" || name == "Encoder<word>") {
    nameOut = name;
    return true;
  }
  return false;
}

// Returns true if `t` is an emitc.array whose element type is a move-only
// opaque. Used by the DPS-lift post-pass to recognise function returns that
// must be lifted to `std::array<T, N>&` out-params (a C array cannot be
// returned by value in C++).
bool isMoveOnlyArray(Type t, StringRef& eltNameOut, int64_t& sizeOut) {
  auto arrayT = dyn_cast<emitc::ArrayType>(t);
  if (!arrayT) return false;
  if (arrayT.getShape().size() != 1) return false;
  StringRef name;
  if (!isMoveOnlyOpaque(arrayT.getElementType(), name)) return false;
  eltNameOut = name;
  sizeOut = arrayT.getShape()[0];
  return true;
}

// Cheddar types map to the textual C++ type that the CHEDDAR library uses.
// Move-only types (Ciphertext/Plaintext/Constant) are *also* mapped to
// `opaque<X>`; local variables wrap them in `lvalue<X>` only at the point of
// declaration (via `emitc.variable`).
class TypeConverterImpl : public TypeConverter {
 public:
  explicit TypeConverterImpl(MLIRContext* ctx) {
    addConversion([](Type t) { return t; });
    addConversion([ctx](cheddar::ParameterType) -> Type {
      return OpaqueType::get(ctx, "Parameter");
    });
    addConversion([ctx](cheddar::ContextType) -> Type {
      return PointerType::get(ctx, OpaqueType::get(ctx, "Context<word>"));
    });
    addConversion([ctx](cheddar::UserInterfaceType) -> Type {
      return PointerType::get(ctx, OpaqueType::get(ctx, "UserInterface<word>"));
    });
    addConversion([ctx](cheddar::EncoderType) -> Type {
      return OpaqueType::get(ctx, "Encoder<word>");
    });
    addConversion([ctx](cheddar::EvkMapType) -> Type {
      return OpaqueType::get(ctx, "EvkMap<word>");
    });
    addConversion([ctx](cheddar::EvalKeyType) -> Type {
      return OpaqueType::get(ctx, "EvaluationKey<word>");
    });
    addConversion([ctx](cheddar::CiphertextType) -> Type {
      return OpaqueType::get(ctx, "Ciphertext<word>");
    });
    addConversion([ctx](cheddar::PlaintextType) -> Type {
      return OpaqueType::get(ctx, "Plaintext<word>");
    });
    addConversion([ctx](cheddar::ConstantType) -> Type {
      return OpaqueType::get(ctx, "Constant<word>");
    });
    // Tensor messages used by encode/decode become std::vector<double>; the
    // bitwidth choice is library-side and matches CHEDDAR's host-side API.
    addConversion([ctx](RankedTensorType) -> Type {
      return OpaqueType::get(ctx, "std::vector<double>");
    });
    // memref<...x!cheddar.*> shows up after bufferization of looped kernels.
    // For move-only element types we convert directly to emitc.array (a
    // fixed-size C array) so the emitted C++ is `T name[N];` -- subscripting
    // returns a reference and avoids any copy of the move-only element.
    //
    // Non-move-only (e.g. float message) memrefs from the client functions are
    // lowered by upstream MemRefToEmitC. Upstream removed its public
    // type-conversion helper (populateMemRefToEmitCTypeConversion); replicate
    // it here -- a static-shape, identity-layout memref converts to a
    // fixed-size emitc.array of the converted element type. Registered first so
    // our move-only-specific conversion below (added later, hence tried first)
    // can defer to it via std::nullopt.
    addConversion([this](MemRefType type) -> std::optional<Type> {
      if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
          type.getRank() == 0 || llvm::is_contained(type.getShape(), 0))
        return std::nullopt;
      Type converted = this->convertType(type.getElementType());
      if (!converted) return Type();
      return Type(emitc::ArrayType::get(type.getShape(), converted));
    });
    addConversion([this](MemRefType type) -> std::optional<Type> {
      Type converted = this->convertType(type.getElementType());
      if (!converted) return Type();
      StringRef name;
      if (isMoveOnlyOpaque(converted, name)) {
        // Only static, rank-1 shapes are supported: emitc.array can't model
        // dynamic extents, and the DPS boundary lift represents these as 1-D
        // `std::array<T, N>` out-params/args. A dynamic-shape move-only memref
        // would otherwise fall through to a memref<...x!emitc.opaque> that
        // stock MemRefToEmitC lowers with descriptors/copies (invalid for
        // move-only payloads), and a higher-rank one to a multi-dim emitc.array
        // the boundary lift can't represent. Return a null type so the
        // conversion fails loudly rather than emitting broken C++.
        if (!type.hasStaticShape() || type.getRank() != 1) return Type();
        return Type(emitc::ArrayType::get(type.getShape(), converted));
      }
      // A strided/offset float memref is a `memref.subview` slice feeding
      // cheddar.encode. Model it as a raw pointer (`float*`) into the base
      // array; ConvertSubViewToPointer materialises `&base[offs...]` and
      // ConvertEncode reads N contiguous elements from it. Identity-layout
      // float memrefs carry no StridedLayoutAttr and fall through to upstream
      // MemRefToEmitC's array conversion.
      if (isa<FloatType>(converted) && isa<StridedLayoutAttr>(type.getLayout()))
        return Type(emitc::PointerType::get(converted));
      // Defer to upstream MemRefToEmitC (registered above).
      return std::nullopt;
    });
    // `index` is what `memref.load`/`memref.store`/`tensor.extract` use as
    // their index operand; SCFToEmitC + ArithToEmitC convert these to
    // `emitc.size_t` and leave a `builtin.unrealized_conversion_cast` at the
    // boundary. Hooking `index -> emitc.size_t` here lets our memref-op
    // patterns consume the converted index directly via the adaptor and the
    // dialect-conversion framework reconciles the cast away.
    addConversion(
        [ctx](IndexType) -> Type { return emitc::SizeTType::get(ctx); });
  }
};

// emitc::OpaqueType doesn't implement MemRefElementTypeInterface upstream,
// which would block our type converter from forming
// `memref<Nx!emitc.opaque<X<word>>>` (the natural converted form of
// `memref<Nx!cheddar.X>`). The interface is marker-only, so an empty external
// model suffices.
struct EmitCOpaqueAsMemRefElement
    : public mlir::MemRefElementTypeInterface::ExternalModel<
          EmitCOpaqueAsMemRefElement, mlir::emitc::OpaqueType> {};

// Generic conversion pattern for memref ops carrying non-move-only element
// types: rebuilds the op with operand/result types converted by `tc`.
// Move-only memref ops are handled by the more specific patterns below.
struct ConvertGenericMemRefOp : public ConversionPattern {
  ConvertGenericMemRefOp(const TypeConverter& tc, MLIRContext* ctx)
      : ConversionPattern(tc, MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (op->getDialect()->getNamespace() != "memref") return failure();
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    OperationState state(op->getLoc(), op->getName(), operands, newResultTypes,
                         op->getAttrs(), op->getSuccessors());
    for ([[maybe_unused]] auto& r : op->getRegions()) {
      return failure();
    }
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// memref.alloc producing a memref of move-only cheddar type -> emitc.variable
// of emitc.array type. Emits `T name[N];` -- N default-constructed elements,
// move-only-safe, RAII-cleaned at scope exit.
struct ConvertMemRefAllocMoveOnly
    : public OpConversionPattern<mlir::memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    auto arrayTy = dyn_cast_or_null<emitc::ArrayType>(converted);
    if (!arrayTy) return failure();
    auto var = emitc::VariableOp::create(
        rewriter, op.getLoc(), arrayTy,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(op, var.getResult());
    return success();
  }
};

// memref.load on an emitc.array of move-only opaque -> emitc.subscript +
// emitc.load. The subscript is alwaysInline=true so emission inlines `m[i]`
// at the use site, and the load is then erased by the move-only load-elision
// post-pass for consumers that accept lvalues directly (cheddar verbatims,
// memref.store-as-verbatim, return).
struct ConvertMemRefLoadMoveOnly
    : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value buf = adaptor.getMemref();
    auto arrayTy = dyn_cast<emitc::ArrayType>(buf.getType());
    if (!arrayTy) return failure();
    StringRef name;
    if (!isMoveOnlyOpaque(arrayTy.getElementType(), name)) return failure();
    auto lvalT = emitc::LValueType::get(arrayTy.getElementType());
    auto sub = emitc::SubscriptOp::create(rewriter, op.getLoc(), lvalT, buf,
                                          adaptor.getIndices());
    rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, arrayTy.getElementType(),
                                               sub.getResult());
    return success();
  }
};

// memref.store of a move-only value into an emitc.array -> subscript + a
// verbatim that emits `arr[i] = std::move(src);`. The load-elision post-pass
// will (correctly) substitute the load on `src` with the underlying lvalue,
// at which point the verbatim's two operands are both lvalues and the
// emission prints two variable names.
struct ConvertMemRefStoreMoveOnly
    : public OpConversionPattern<mlir::memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value buf = adaptor.getMemref();
    auto arrayTy = dyn_cast<emitc::ArrayType>(buf.getType());
    if (!arrayTy) return failure();
    StringRef name;
    if (!isMoveOnlyOpaque(arrayTy.getElementType(), name)) return failure();
    auto lvalT = emitc::LValueType::get(arrayTy.getElementType());
    auto sub = emitc::SubscriptOp::create(rewriter, op.getLoc(), lvalT, buf,
                                          adaptor.getIndices());
    emitc::VerbatimOp::create(rewriter, op.getLoc(), "{} = std::move({});",
                              ValueRange{sub.getResult(), adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.dealloc on a move-only emitc.array is a no-op at emission (the
// scope-bound emitc.variable's destructor handles cleanup). Erase the op.
struct EraseMemRefDeallocMoveOnly
    : public OpConversionPattern<mlir::memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::DeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value buf = adaptor.getMemref();
    auto arrayTy = dyn_cast<emitc::ArrayType>(buf.getType());
    if (!arrayTy) return failure();
    StringRef name;
    if (!isMoveOnlyOpaque(arrayTy.getElementType(), name)) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.subview producing a strided slice of a *float* buffer -> a raw
// pointer into the converted base array: `&base[o0][o1]...`. After
// bufferization + fold-memref-alias-ops the only residual subviews are the
// per-weight slices feeding `cheddar.encode`; the slice is contiguous (a full
// row of the packed weight buffer), so ConvertEncode reads N elements from the
// pointer. Move-only (ciphertext/plaintext) memrefs never reach here -- they
// are rank-1 emitc.array handled by the move-only patterns above.
struct ConvertSubViewToPointer
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    auto arrayTy = dyn_cast<emitc::ArrayType>(base.getType());
    if (!arrayTy || !isa<FloatType>(arrayTy.getElementType())) return failure();
    // Only static offsets (the bufferized weight-packing slices are static),
    // and one offset per source dimension so the subscript is fully indexed.
    auto offsets = op.getStaticOffsets();
    if (static_cast<int64_t>(offsets.size()) != arrayTy.getShape().size())
      return failure();
    for (int64_t o : offsets)
      if (ShapedType::isDynamic(o)) return failure();
    auto sizeT = emitc::SizeTType::get(getContext());
    SmallVector<Value> idx;
    for (int64_t o : offsets)
      idx.push_back(emitc::LiteralOp::create(rewriter, op.getLoc(), sizeT,
                                             std::to_string(o)));
    auto lvalT = emitc::LValueType::get(arrayTy.getElementType());
    auto sub =
        emitc::SubscriptOp::create(rewriter, op.getLoc(), lvalT, base, idx);
    rewriter.replaceOpWithNewOp<emitc::AddressOfOp>(
        op, emitc::PointerType::get(arrayTy.getElementType()), sub.getResult());
    return success();
  }
};

// Like upstream MemRefToEmitC's ConvertGlobal but tolerates an alignment
// attribute: bufferization stamps the constant weight/bias globals with
// `alignment = 64`, on which the upstream pattern bails. emitc.global has no
// `alignas` specifier so the alignment is simply dropped. Benefit 2 wins over
// the upstream (benefit 1) pattern.
struct ConvertGlobalDropAlign
    : public OpConversionPattern<mlir::memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().hasStaticShape()) return failure();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) return failure();
    auto vis = SymbolTable::getSymbolVisibility(op);
    if (vis != SymbolTable::Visibility::Public &&
        vis != SymbolTable::Visibility::Private)
      return failure();
    bool staticSpecifier = vis == SymbolTable::Visibility::Private;
    Attribute initialValue = adaptor.getInitialValueAttr();
    if (isa_and_present<UnitAttr>(initialValue)) initialValue = {};
    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, adaptor.getSymName(), resultTy, initialValue,
        /*externSpecifier=*/!staticSpecifier, staticSpecifier,
        adaptor.getConstant());
    return success();
  }
};

// memref.copy between float buffers -> an element-wise loop. After
// ConvertSubViewToPointer the (strided slice) operands are raw `float*`; the
// slices are contiguous full rows so copying N elements by linear index is
// correct. (Upstream MemRefToEmitC has no copy pattern.)
struct ConvertMemRefCopy : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value src = adaptor.getSource();
    Value tgt = adaptor.getTarget();
    auto floatOperand = [](Value v) -> bool {
      if (auto p = dyn_cast<emitc::PointerType>(v.getType()))
        return isa<FloatType>(p.getPointee());
      if (auto a = dyn_cast<emitc::ArrayType>(v.getType()))
        return isa<FloatType>(a.getElementType());
      return false;
    };
    if (!floatOperand(src) || !floatOperand(tgt)) return failure();
    int64_t n = 1;
    if (auto sh = dyn_cast<ShapedType>(op.getSource().getType()))
      for (int64_t d : sh.getShape()) n *= d;
    // Parenthesised pointer to the first element, flat-indexable as `(b)[_i]`.
    // A pointer operand already is that; an emitc.array needs `&arr[0]..[0]`
    // (one `[0]` per dim) -- `&arr[0]` alone is a row pointer, not the element.
    auto begin = [](Value v) -> std::string {
      if (isa<emitc::PointerType>(v.getType())) return "({})";
      auto a = cast<emitc::ArrayType>(v.getType());
      std::string s = "(&{}";
      for (size_t i = 0; i < a.getShape().size(); ++i) s += "[0]";
      return s + ")";
    };
    // Single `for` statement, no enclosing block: `_i` is scoped to the loop,
    // so no name clash with other copies and no literal braces to escape.
    std::string fmt = "for (size_t _i = 0; _i < " + std::to_string(n) +
                      "; ++_i) " + begin(tgt) + "[_i] = " + begin(src) +
                      "[_i];";
    VerbatimOp::create(rewriter, op.getLoc(), fmt, ValueRange{tgt, src});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.decode is DPS: it writes the decoded message into a float buffer
// (the bufferized `value`/result memref -> emitc.array). CHEDDAR's
// `Decode(std::vector<Complex>& out, const Plaintext& pt)` returns a complex
// vector, so decode into a temporary and copy the real parts into the buffer.
struct ConvertDecode : public OpConversionPattern<cheddar::DecodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::DecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value dst = adaptor.getValue();
    auto arrayTy = dyn_cast<emitc::ArrayType>(dst.getType());
    if (!arrayTy || !isa<FloatType>(arrayTy.getElementType())) return failure();
    auto shape = arrayTy.getShape();
    std::string idxPrefix;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      if (shape[i] != 1) return failure();  // only <1x...xN> result buffers
      idxPrefix += "[0]";
    }
    // Let emitc allocate the temporary message vector (unique name, scoped) --
    // no hand-written block, so no literal braces to escape.
    auto* ctx = rewriter.getContext();
    Value vec = VariableOp::create(
        rewriter, op.getLoc(),
        LValueType::get(OpaqueType::get(ctx, "std::vector<Complex>")),
        OpaqueAttr::get(ctx, ""));
    VerbatimOp::create(
        rewriter, op.getLoc(), "{}.Decode({}, {});",
        ValueRange{adaptor.getEncoder(), vec, adaptor.getPlaintext()});
    VerbatimOp::create(rewriter, op.getLoc(),
                       "for (size_t _i = 0; _i < " +
                           std::to_string(shape.back()) + "; ++_i) {}" +
                           idxPrefix + "[_i] = {}[_i].real();",
                       ValueRange{dst, vec});
    rewriter.replaceOp(op, dst);
    return success();
  }
};

// Declare a fresh local lvalue variable of value-type `t`. Patterns then emit
// a verbatim that initializes it (out-param call or assignment from a getter)
// and finish with `loadAfter` to obtain the loaded value to feed into
// `replaceOp`, matching the type converter's value-form output.
Value declareLocal(OpBuilder& b, Location loc, Type t) {
  return VariableOp::create(b, loc, LValueType::get(t),
                            OpaqueAttr::get(b.getContext(), ""));
}

Value loadAfter(OpBuilder& b, Location loc, Type t, Value lvalue) {
  return mlir::emitc::LoadOp::create(b, loc, t, lvalue);
}

// Emit `receiver.method(out, args..., extra)` -- or `receiver->method(...)` for
// a pointer receiver. `emitc.member_call_opaque` picks `.`/`->` from the
// receiver type, so no manual pointer check is needed. Any trailing literals
// (`extra`, e.g. a level/scale/bool) are appended as one opaque constant arg.
void emitOutParamCall(OpBuilder& b, Location loc, Value receiver,
                      StringRef method, Value out, ValueRange args,
                      StringRef extra = "") {
  SmallVector<Value> argOperands{out};
  argOperands.append(args.begin(), args.end());
  ArrayAttr argsAttr;
  if (!extra.empty()) {
    // `args` interleaves operands (by index) and constant attrs: list the arg
    // operands in order, then the trailing literal(s) as one opaque constant.
    SmallVector<Attribute> a;
    for (size_t i = 0; i < argOperands.size(); ++i)
      a.push_back(b.getIndexAttr(i));
    a.push_back(emitc::OpaqueAttr::get(b.getContext(), extra));
    argsAttr = b.getArrayAttr(a);
  }
  MemberCallOpaqueOp::create(b, loc, /*resultTypes=*/TypeRange{}, receiver,
                             b.getStringAttr(method), argsAttr,
                             /*template_args=*/ArrayAttr{}, argOperands);
}

std::string intLit(IntegerAttr a) { return std::to_string(a.getInt()); }

std::string floatLit(FloatAttr a) {
  // Emit full double precision (%.17g round-trips exactly). The default
  // formatv precision truncates, which silently corrupts literal scales (e.g.
  // a scale-alignment factor like 1.0000000744039537 collapsing to 1.00).
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", a.getValueAsDouble());
  return std::string(buf);
}

// `{}` is the operand placeholder in `emitc.verbatim` format strings. A
// literal `{` must be written `{{`; a literal `}` is emitted as-is (emitc does
// NOT collapse `}}` to `}`), so the closing brace of an initializer list stays
// single -- doubling it would emit a stray `}`.
std::string i32ArrayLit(DenseI32ArrayAttr a) {
  std::string s = "{{";
  for (size_t i = 0; i < a.size(); ++i) {
    if (i > 0) s += ", ";
    s += std::to_string(a[i]);
  }
  return s + "}";
}

std::string floatArrayLit(ArrayAttr a) {
  std::string s = "{{";
  for (size_t i = 0; i < a.size(); ++i) {
    if (i > 0) s += ", ";
    s +=
        llvm::formatv("{0:f1}", cast<FloatAttr>(a[i]).getValueAsDouble()).str();
  }
  return s + "}";
}

// Generic out-param pattern: first operand is the receiver, remaining operands
// are inputs, the op produces one cheddar value.
template <typename Op>
struct OutParamPattern : public OpConversionPattern<Op> {
  OutParamPattern(const TypeConverter& tc, MLIRContext* ctx, StringRef method)
      : OpConversionPattern<Op>(tc, ctx), method(method.str()) {}

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    auto operands = adaptor.getOperands();
    emitOutParamCall(rewriter, op.getLoc(), operands[0], method, out,
                     operands.drop_front());
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }

  std::string method;
};

// CreateContext: static factory, rendered as `T x = T::Create(args);`.
struct ConvertCreateContext
    : public OpConversionPattern<cheddar::CreateContextOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::CreateContextOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    auto call = CallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{t},
        rewriter.getStringAttr("Context<word>::Create"),
        ValueRange{adaptor.getParams()},
        /*args=*/ArrayAttr{}, /*template_args=*/ArrayAttr{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// The getter-style setup ops -- get_evk_map, get_mult_key, get_encoder,
// create_user_interface -- are NOT supported by this lowering. Their CHEDDAR
// C++ counterparts hand back a `const&` to a move-only / non-assignable value
// (EvkMap and EvaluationKey are move-only; Encoder is a view with reference
// members; UserInterface is move-only), so the value-materialising shape a
// naive lowering emits -- `T tmp; tmp = recv->Get();` -- cannot compile.
// Supporting them needs inline-at-use emission (the way HRot already inlines
// `ui->GetRotationKey(d)`) or emitc.expression. Real kernels avoid these
// entirely: they take keys/maps/encoders as function arguments or look keys
// up inline.
//
// Reject them with a clear diagnostic in a pre-pass walk rather than from a
// conversion pattern: the dialect-conversion framework discards diagnostics
// emitted by a pattern that returns failure, so a pattern-based error would be
// swallowed in favour of a generic "failed to legalize". Returns true (and
// emits an error on each) if any unsupported getter is present.
bool diagnoseUnsupportedGetters(Operation* root) {
  bool found = false;
  root->walk([&](Operation* op) {
    if (isa<cheddar::GetEvkMapOp, cheddar::GetMultKeyOp, cheddar::GetEncoderOp,
            cheddar::CreateUserInterfaceOp>(op)) {
      op->emitError()
          << "cheddar-to-emitc: lowering of '" << op->getName().getStringRef()
          << "' is not supported: it returns a const reference to a "
             "move-only/non-assignable value, which cannot be materialised "
             "into a local without a copy. Pass the key/map/encoder as a "
             "function argument, or look it up inline at the use site.";
      found = true;
    }
  });
  return found;
}

// An scf op that carries a move-only Cheddar payload as a loop-carried /
// result value cannot be lowered. SCFToEmitC materialises each result as an
// `emitc.variable` initialised by an `emitc.assign` from the init value
// (`acc = init;`); for a move-only payload that is a deleted copy assignment
// (the init is a `const T&` input after boundary tightening). Realistic
// reduction kernels avoid this by writing into a pre-existing accumulator or
// output buffer (the value flows through a memref, not through iter_args), so
// reject the iter_args shape up front with a clear diagnostic rather than
// emitting C++ that fails to compile. Returns true if any such op is present.
bool diagnoseMoveOnlyLoopCarry(Operation* root) {
  auto isMoveOnlyCheddar = [](Type t) {
    return isa<cheddar::CiphertextType, cheddar::PlaintextType,
               cheddar::ConstantType, cheddar::EvalKeyType>(t);
  };
  bool found = false;
  root->walk([&](Operation* op) {
    if (!isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::IfOp,
             mlir::scf::IndexSwitchOp>(op))
      return;
    for (Type t : op->getResultTypes()) {
      if (!isMoveOnlyCheddar(t)) continue;
      op->emitError()
          << "cheddar-to-emitc: '" << op->getName().getStringRef()
          << "' carrying a move-only Cheddar value through a loop-carried or "
             "result value is not supported: lowering would copy-initialise an "
             "accumulator from the value, but the payload types are move-only. "
             "Write into a pre-existing accumulator or output buffer instead.";
      found = true;
      break;
    }
  });
  return found;
}

struct ConvertPrepareRotKey
    : public OpConversionPattern<cheddar::PrepareRotKeyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::PrepareRotKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string extra =
        intLit(op.getDistanceAttr()) + ", " + intLit(op.getMaxLevelAttr());
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->PrepareRotationKey(" + extra + ");",
                       ValueRange{adaptor.getUi()});
    rewriter.eraseOp(op);
    return success();
  }
};

// The enclosing function's Context arg (threaded by LWEToCheddar), used to read
// CHEDDAR's exact canonical per-level scale. Null if absent (e.g. emitter unit
// tests with no context arg).
Value findCtx(Operation* op, const TypeConverter& tc) {
  Type ctxType = tc.convertType(cheddar::ContextType::get(op->getContext()));
  auto r = getContextualArgFromFunc(op, ctxType);
  if (failed(r)) return Value{};
  return r.value();
}

struct ConvertEncode : public OpConversionPattern<cheddar::EncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    std::string lvl = std::to_string(op.getLevelAttr().getInt());

    // The message is a raw `float*` into the (converted) base array -- a
    // `memref.subview` slice lowered by ConvertSubViewToPointer. CHEDDAR's
    // `Encode(pt, level, scale, msg)` takes a `std::vector<Complex>`, so wrap
    // the N contiguous message elements in one (each float -> Complex(re, 0)).
    Value msg = adaptor.getMessage();
    int64_t n = 1;
    if (auto sh = dyn_cast<ShapedType>(op.getMessage().getType()))
      for (int64_t d : sh.getShape()) n *= d;
    std::string begin =
        isa<emitc::PointerType>(msg.getType()) ? "{}" : "&{}[0]";

    // CHEDDAR keeps an EXACT canonical scale per level (rescale divides by the
    // exact prime product, so the scale drifts from a clean 2^k) and rejects
    // any mismatch beyond 1e-12. So we must hand the encoder CHEDDAR's own
    // `ctx->param_.GetScale(level)` rather than a literal 2^k. For a doubled
    // (post-mult, pre-rescale) plaintext the scale is GetScale(level) raised to
    // logScale/logDefaultScale.
    Value ctx = findCtx(op, *this->typeConverter);
    int64_t logDefaultScale = 0;
    if (auto m = op->getParentOfType<ModuleOp>())
      if (auto a = m->getAttrOfType<IntegerAttr>("cheddar.logDefaultScale"))
        logDefaultScale = a.getInt();

    std::string scaleExpr;
    SmallVector<Value> scaleOperands;
    // Opt-in exact (drifted) scale: a pre-rescale plaintext (e.g. a relu bias)
    // whose ciphertext scale has diverged from canonical GetScale^k by exact
    // rescale-prime factors. Emitted verbatim. Hand-supplied for the MNIST
    // biases (HACK #7); precise scale management would compute it.
    if (auto exact = op->getAttrOfType<FloatAttr>("cheddar.exact_scale")) {
      scaleExpr = floatLit(exact);
    } else if (ctx && logDefaultScale > 0) {
      int64_t logScale = static_cast<int64_t>(
          std::llround(std::log2(op.getScaleAttr().getValueAsDouble())));
      int64_t mult =
          (logScale >= logDefaultScale && logScale % logDefaultScale == 0)
              ? logScale / logDefaultScale
              : 1;
      for (int64_t i = 0; i < mult; ++i) {
        if (i) scaleExpr += " * ";
        scaleExpr += "{}->param_.GetScale(" + lvl + ")";
        scaleOperands.push_back(ctx);
      }
    } else {
      scaleExpr = floatLit(op.getScaleAttr());
    }

    // Let emitc allocate the message vector (unique name, scoped); fill it
    // from the float range, then call Encode. All brace-free statements -- no
    // hand-written block, so no literal braces to escape in the verbatims.
    Value vec = declareLocal(
        rewriter, op.getLoc(),
        OpaqueType::get(rewriter.getContext(), "std::vector<Complex>"));
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{} = std::vector<Complex>(" + begin + ", " + begin +
                           " + " + std::to_string(n) + ");",
                       ValueRange{vec, msg, msg});
    // Sequential `{}`: encoder, out, one ctx per GetScale factor, then vec.
    SmallVector<Value> operands{adaptor.getEncoder(), out};
    operands.append(scaleOperands.begin(), scaleOperands.end());
    operands.push_back(vec);
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}.Encode({}, " + lvl + ", " + scaleExpr + ", {});",
                       operands);

    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertEncodeConstant
    : public OpConversionPattern<cheddar::EncodeConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    // CHEDDAR signature: encoder.EncodeConstant(constant, level, scale,
    // number). The scalar `number` (the value) is the LAST argument, after the
    // level and scale literals -- emitOutParamCall cannot place an SSA operand
    // after the trailing literals, so emit the call directly in the correct
    // order.
    std::string fmt = "{}.EncodeConstant({}, " + intLit(op.getLevelAttr()) +
                      ", " + floatLit(op.getScaleAttr()) + ", {});";
    VerbatimOp::create(
        rewriter, op.getLoc(), fmt,
        ValueRange{adaptor.getEncoder(), out, adaptor.getValue()});
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertLevelDown : public OpConversionPattern<cheddar::LevelDownOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::LevelDownOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    emitOutParamCall(rewriter, op.getLoc(), adaptor.getCtx(), "LevelDown", out,
                     ValueRange{adaptor.getInput()},
                     intLit(op.getTargetLevelAttr()));
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertHMult : public OpConversionPattern<cheddar::HMultOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HMultOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    StringRef extra = op.getRescale() ? "true" : "false";
    emitOutParamCall(
        rewriter, op.getLoc(), adaptor.getCtx(), "HMult", out,
        ValueRange{adaptor.getLhs(), adaptor.getRhs(), adaptor.getMultKey()},
        extra);
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

// HRot/HRotAdd/HConj/HConjAdd: the dialect no longer carries an explicit
// key SSA operand. At emission time, we look up `ui->GetRotationKey(d)` (or
// `ui->GetConjugationKey()`) inline; the UserInterface is taken from the
// enclosing function's argument list via `getContextualArgFromFunc`.
//
// `getContextualArgFromFunc` walks the original (pre-conversion) func block
// and looks for an arg of the *converted* type; the pass runs as a partial
// dialect conversion so the func signature has already been converted by the
// structural patterns when these matchers fire.

Value findUi(Operation* op, const TypeConverter& tc) {
  Type uiType =
      tc.convertType(cheddar::UserInterfaceType::get(op->getContext()));
  auto r = getContextualArgFromFunc(op, uiType);
  if (failed(r)) return Value{};
  return r.value();
}

struct ConvertHRot : public OpConversionPattern<cheddar::HRotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HRotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = findUi(op, *this->typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");

    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);

    if (auto sd = op.getStaticDistanceAttr()) {
      std::string d = intLit(sd);
      VerbatimOp::create(
          rewriter, op.getLoc(),
          "{}->HRot({}, {}, {}->GetRotationKey(" + d + "), " + d + ");",
          ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui});
    } else {
      Value dyn = adaptor.getDynamicDistance();
      VerbatimOp::create(
          rewriter, op.getLoc(),
          "{}->HRot({}, {}, {}->GetRotationKey({}), {});",
          ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui, dyn, dyn});
    }
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertHRotAdd : public OpConversionPattern<cheddar::HRotAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HRotAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = findUi(op, *this->typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    std::string d = intLit(op.getDistanceAttr());
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{}->HRotAdd({}, {}, {}, {}->GetRotationKey(" + d + "), " + d + ");",
        ValueRange{adaptor.getCtx(), out, adaptor.getInput(),
                   adaptor.getAddend(), ui});
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertHConj : public OpConversionPattern<cheddar::HConjOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = findUi(op, *this->typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    VerbatimOp::create(
        rewriter, op.getLoc(), "{}->HConj({}, {}, {}->GetConjugationKey());",
        ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui});
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertHConjAdd : public OpConversionPattern<cheddar::HConjAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = findUi(op, *this->typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->HConjAdd({}, {}, {}, {}->GetConjugationKey());",
                       ValueRange{adaptor.getCtx(), out, adaptor.getInput(),
                                  adaptor.getAddend(), ui});
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

// `ctx->MadUnsafe(acc, in, c);` is an in-place mutation: the SSA result is
// the same value as the input accumulator after the call.
struct ConvertMadUnsafe : public OpConversionPattern<cheddar::MadUnsafeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::MadUnsafeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    VerbatimOp::create(rewriter, op.getLoc(), "{}->MadUnsafe({}, {}, {});",
                       ValueRange{adaptor.getCtx(), adaptor.getAccumulator(),
                                  adaptor.getInput(), adaptor.getConstant()});
    rewriter.replaceOp(op, adaptor.getAccumulator());
    return success();
  }
};

// `ctx->Boot(res, input, evk_map);`. The CHEDDAR runtime resolves whether
// `ctx` is a regular `Context` or a `BootContext`; the dialect carries only
// a single Context type today.
struct ConvertBoot : public OpConversionPattern<cheddar::BootOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::BootOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    emitOutParamCall(rewriter, op.getLoc(), adaptor.getCtx(), "Boot", out,
                     ValueRange{adaptor.getInput(), adaptor.getEvkMap()});
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertLinearTransform
    : public OpConversionPattern<cheddar::LinearTransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    std::string extra = i32ArrayLit(op.getDiagonalIndicesAttr()) + ", " +
                        intLit(op.getLevelAttr()) + ", " +
                        intLit(op.getLogBabyStepGiantStepRatioAttr());
    emitOutParamCall(rewriter, op.getLoc(), adaptor.getCtx(), "LinearTransform",
                     out,
                     ValueRange{adaptor.getInput(), adaptor.getEvkMap(),
                                adaptor.getDiagonals()},
                     extra);
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct ConvertEvalPoly : public OpConversionPattern<cheddar::EvalPolyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EvalPolyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = this->typeConverter->convertType(op.getResult().getType());
    Value out = declareLocal(rewriter, op.getLoc(), t);
    std::string extra = floatArrayLit(op.getCoefficientsAttr()) + ", " +
                        intLit(op.getLevelAttr());
    emitOutParamCall(rewriter, op.getLoc(), adaptor.getCtx(), "EvalPoly", out,
                     ValueRange{adaptor.getInput(), adaptor.getEvkMap()},
                     extra);
    rewriter.replaceOp(op, loadAfter(rewriter, op.getLoc(), t, out));
    return success();
  }
};

struct CheddarToEmitCPass
    : public impl::CheddarToEmitCBase<CheddarToEmitCPass> {
  using CheddarToEmitCBase::CheddarToEmitCBase;

  void runOnOperation() override {
    auto* ctx = &getContext();

    // Reject unsupported getter-style setup ops before conversion (see
    // diagnoseUnsupportedGetters).
    if (diagnoseUnsupportedGetters(getOperation()) ||
        diagnoseMoveOnlyLoopCarry(getOperation())) {
      signalPassFailure();
      return;
    }

    // A move-only memref arg that is *stored into* is an output/in-place
    // buffer: at the C++ boundary it must become a mutable `std::array<T, N>&`,
    // not the `const std::array<T, N>&` that a read-only input array gets.
    // memref.store ops are gone after conversion (rewritten to subscript + a
    // std::move verbatim), so capture which entry-block-arg memrefs are written
    // *now*, keyed by func op (whose identity persists through the in-place
    // signature conversion). Consumed by Pass 1 below.
    llvm::DenseMap<Operation*, llvm::SmallDenseSet<unsigned>> writtenArrayArgs;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal()) return;
      Block& entry = fn.getBody().front();
      fn.walk([&](mlir::memref::StoreOp store) {
        if (auto ba = dyn_cast<BlockArgument>(store.getMemRef()))
          if (ba.getOwner() == &entry)
            writtenArrayArgs[fn.getOperation()].insert(ba.getArgNumber());
      });
    });

    TypeConverterImpl tc(ctx);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<cheddar::CheddarDialect>();
    target.addLegalDialect<::mlir::emitc::EmitCDialect>();
    target.addLegalDialect<::mlir::func::FuncDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return tc.isSignatureLegal(op.getFunctionType()) &&
             tc.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp, func::CallOp>(
        [&](Operation* op) { return tc.isLegal(op); });
    // memref ops carrying Cheddar element types after bufferization of looped
    // kernels are legal only once their element types have been converted
    // (Cheddar element type -> EmitC opaque). The recursive MemRefType
    // converter in TypeConverterImpl plus the structural conversion patterns
    // do the rewrite.
    target.addDynamicallyLegalDialect<::mlir::memref::MemRefDialect>(
        [&](Operation* op) { return tc.isLegal(op); });
    // memref.global (constant weight/bias arrays) has no typed operand/result,
    // so the dialect-level `tc.isLegal` check above always considers it legal
    // and never converts it -- leaving the converted emitc.get_global dangling
    // ("does not reference a valid emitc.global"). Force it illegal so upstream
    // ConvertGlobal lowers it to emitc.global.
    target.addIllegalOp<::mlir::memref::GlobalOp>();
    RewritePatternSet patterns(ctx);
    addStructuralConversionPatterns(tc, patterns, target);
    mlir::populateArithToEmitCPatterns(tc, patterns);
    mlir::populateSCFToEmitCConversionPatterns(patterns, tc);
    // Lower the client functions' non-move-only (float message) memrefs with
    // upstream MemRefToEmitC (alloc/load/store/global). The residual strided
    // `memref.subview` slices feeding cheddar.encode are lowered to raw
    // pointers by ConvertSubViewToPointer (registered below).
    mlir::populateMemRefToEmitCConversionPatterns(patterns, tc);
    // SCF/Arith from bufferized loop kernels are lowered to EmitC in this same
    // conversion (rather than as separate downstream passes) so the shared type
    // converter sees Cheddar types: an `scf.for` carrying an `!cheddar.*`
    // iter_arg lowers to an `emitc.variable` mutated by move-assignment, and
    // loop-index `arith`/`index` values feed `memref` subscripts without
    // stranding an `index -> emitc.size_t` cast at a pass boundary.
    //
    // These op-level illegality markings must come *after*
    // addStructuralConversionPatterns: that helper calls
    // populateSCFStructuralTypeConversionsAndLegality, which marks the SCF ops
    // dynamically legal (type-conversion only). Op-level legality overrides the
    // dialect-level setting regardless of call order, so a dialect-level
    // addIllegalDialect<SCFDialect> would be silently shadowed and the EmitC
    // for/if patterns would never fire. Marking the ops illegal here forces the
    // SCFToEmitC patterns to win.
    target.addIllegalOp<::mlir::scf::ForOp, ::mlir::scf::IfOp,
                        ::mlir::scf::IndexSwitchOp>();
    target.addIllegalDialect<::mlir::arith::ArithDialect>();
    // Memref ops carrying cheddar element types: move-only-aware patterns
    // emit emitc.variable/subscript/verbatim directly (handling C++ move
    // semantics correctly). The generic fallback rebuilds non-move-only
    // memref ops with converted types for downstream memref-to-emitc to
    // lower the standard way.
    // Higher benefit than the upstream MemRefToEmitC patterns so the move-only
    // (ciphertext/plaintext) memref ops win here; these patterns decline for
    // non-move-only element types, so float message memrefs fall through to
    // upstream MemRefToEmitC. (ConvertGenericMemRefOp is gone: upstream
    // MemRefToEmitC is the real downstream lowering it used to defer to.)
    patterns.add<ConvertMemRefAllocMoveOnly, ConvertMemRefLoadMoveOnly,
                 ConvertMemRefStoreMoveOnly, EraseMemRefDeallocMoveOnly,
                 ConvertSubViewToPointer, ConvertMemRefCopy,
                 ConvertGlobalDropAlign>(tc, ctx, /*benefit=*/2);

    // get_evk_map / get_mult_key / get_encoder / create_user_interface are
    // rejected up front by diagnoseUnsupportedGetters, so no patterns are
    // registered for them here.
    patterns.add<ConvertCreateContext, ConvertPrepareRotKey, ConvertEncode,
                 ConvertDecode, ConvertEncodeConstant, ConvertLevelDown,
                 ConvertHMult, ConvertHRot, ConvertHRotAdd, ConvertHConj,
                 ConvertHConjAdd, ConvertMadUnsafe, ConvertBoot,
                 ConvertLinearTransform, ConvertEvalPoly>(tc, ctx);

    // The remaining ops follow the uniform out-param pattern; first operand
    // is the receiver, remaining operands are inputs.
    patterns.add<OutParamPattern<cheddar::AddOp>>(tc, ctx, "Add");
    patterns.add<OutParamPattern<cheddar::SubOp>>(tc, ctx, "Sub");
    patterns.add<OutParamPattern<cheddar::MultOp>>(tc, ctx, "Mult");
    // CHEDDAR's Context uses C++ overloading for ct+pt / ct+const variants:
    // `void Add(Ct&, const Ct&, const Pt&)`, `void Add(Ct&, const Ct&, const
    // Const&)`, etc.  No separate `AddPlain`/`AddConst` methods exist on
    // Context, so dispatch the dialect's `*_plain` / `*_const` ops to the
    // base name and let the C++ compiler pick the overload by arg type.
    patterns.add<OutParamPattern<cheddar::AddPlainOp>>(tc, ctx, "Add");
    patterns.add<OutParamPattern<cheddar::SubPlainOp>>(tc, ctx, "Sub");
    patterns.add<OutParamPattern<cheddar::MultPlainOp>>(tc, ctx, "Mult");
    patterns.add<OutParamPattern<cheddar::AddConstOp>>(tc, ctx, "Add");
    patterns.add<OutParamPattern<cheddar::MultConstOp>>(tc, ctx, "Mult");
    patterns.add<OutParamPattern<cheddar::NegOp>>(tc, ctx, "Neg");
    patterns.add<OutParamPattern<cheddar::RescaleOp>>(tc, ctx, "Rescale");
    patterns.add<OutParamPattern<cheddar::RelinearizeOp>>(tc, ctx,
                                                          "Relinearize");
    patterns.add<OutParamPattern<cheddar::RelinearizeRescaleOp>>(
        tc, ctx, "RelinearizeRescale");
    patterns.add<OutParamPattern<cheddar::EncryptOp>>(tc, ctx, "Encrypt");
    patterns.add<OutParamPattern<cheddar::DecryptOp>>(tc, ctx, "Decrypt");

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    // CHEDDAR's Ciphertext/Plaintext/Constant are move-only in C++.  The
    // structural conversion above leaves func.func ops with `T` (by-value)
    // arg and result types, which the C++ emitter renders as
    //     T add_kernel(..., T a, T b) { T r; ctx->Add(r, a, b); T tmp = r;
    //                                   return tmp; }
    // — both the by-value parameters and the `T tmp = r;` copy-init at the
    // return rely on a copy ctor that doesn't exist.
    //
    // Lift each such func to destination-passing style at the EmitC level:
    //   * move-only arg types become `const T&`,
    //   * move-only result types are dropped and re-appended as trailing
    //     `T&` out-params,
    //   * each `func.return %r` becomes `out = std::move(<src>);` (using the
    //     load's lvalue source when possible to avoid materialising the
    //     `T tmp = r;` copy) followed by a bare `func.return`.
    //
    // Local variables and intermediate values inside the body stay as plain
    // `T` lvalues — those go through the move-assignment in `ctx->Op(out,
    // ...)` which is valid.
    // The patterns above use the `T tmp; ctx->Op(tmp, ...); emitc.load(tmp)`
    // shape, where the `emitc.load` is intended to materialise a value-typed
    // SSA def the next pattern can consume.  In C++ this renders as
    // `T downstream = tmp;` -- copy-init, which doesn't compile for move-only
    // types.  Walk the IR and elide every load of a move-only opaque,
    // replacing all uses of the load's result with the source lvalue
    // directly.  Downstream consumers are emitc.verbatim ops (and, after the
    // DPS lift below, func.return rewrites), both of which only ever print
    // the operand name -- so the resulting C++ ends up referencing the
    // local variable by name, binding naturally to the `const T&` parameters
    // of the receiver methods.
    getOperation()->walk([](emitc::LoadOp load) {
      StringRef name;
      if (!isMoveOnlyOpaque(load.getType(), name)) return;
      load.getResult().replaceAllUsesWith(load.getOperand());
      load.erase();
    });

    auto& ctxRef = *ctx;
    getOperation()->walk([&ctxRef, &writtenArrayArgs](func::FuncOp op) {
      if (op.isExternal()) return;
      auto funcType = op.getFunctionType();
      Block& entry = op.getBody().front();

      SmallVector<func::ReturnOp> returns;
      op.walk([&](func::ReturnOp r) { returns.push_back(r); });

      // A move-only payload result whose return operand is an entry block
      // argument means the function hands back storage it was given: e.g.
      // `mad_unsafe` mutates its accumulator argument in place and returns it,
      // or a passthrough returns an argument unchanged. Such an argument is
      // the destination -- it must be a mutable `T&` (not `const T&`) so the
      // in-place mutation binds, and it needs no separate out-param. Detect
      // these first so Pass 1 tightens them correctly. Only treated as
      // in-place when *every* return agrees, so a signature is never left
      // inconsistent.
      unsigned numResults = funcType.getNumResults();
      SmallVector<bool> resultIsInout(numResults, false);
      SmallVector<bool> argIsInout(funcType.getNumInputs(), false);
      for (unsigned i = 0; i < numResults; ++i) {
        StringRef name;
        if (!isMoveOnlyOpaque(funcType.getResult(i), name)) continue;
        int argIdx = -1;
        bool ok = !returns.empty();
        for (auto ret : returns) {
          auto ba = dyn_cast<BlockArgument>(ret.getOperand(i));
          if (!ba || ba.getOwner() != &entry) {
            ok = false;
            break;
          }
          if (argIdx < 0)
            argIdx = ba.getArgNumber();
          else if (argIdx != static_cast<int>(ba.getArgNumber())) {
            ok = false;
            break;
          }
        }
        if (ok && argIdx >= 0) {
          resultIsInout[i] = true;
          argIsInout[argIdx] = true;
        }
      }

      // Pass 1: tighten input types at the C++ boundary. Move-only payload
      // scalars become `const T&`, except in-place/returned args which stay
      // mutable `T&`. emitc.array<NxC> args (from memref<NxC> boundary
      // lowering of read-only inputs) become `const std::array<C<word>, N>&`.
      // EvkMap/Encoder become `const T&`. emitc.subscript on the original
      // emitc.array stays valid after the swap because subscript also accepts
      // EmitC_OpaqueType as its base operand.
      SmallVector<Type> newInputs(funcType.getInputs().begin(),
                                  funcType.getInputs().end());
      bool inputsChanged = false;
      for (size_t i = 0; i < newInputs.size(); ++i) {
        StringRef name;
        int64_t arraySize = 0;
        if (isMoveOnlyOpaque(newInputs[i], name)) {
          std::string typeName = argIsInout[i] ? (name + "&").str()
                                               : ("const " + name + "&").str();
          newInputs[i] = emitc::OpaqueType::get(&ctxRef, typeName);
          entry.getArgument(i).setType(newInputs[i]);
          inputsChanged = true;
        } else if (isMoveOnlyArray(newInputs[i], name, arraySize)) {
          // A written (output/in-place) array buffer must be a mutable
          // reference; a read-only input array stays `const`.
          // `subscript`-then-`std::move` assignment into a `const std::array&`
          // would not compile.
          auto it = writtenArrayArgs.find(op.getOperation());
          bool written = it != writtenArrayArgs.end() && it->second.contains(i);
          std::string arrayTy =
              ("std::array<" + name + ", " + std::to_string(arraySize) + ">&")
                  .str();
          std::string typeName = written ? arrayTy : ("const " + arrayTy);
          newInputs[i] = emitc::OpaqueType::get(&ctxRef, typeName);
          entry.getArgument(i).setType(newInputs[i]);
          inputsChanged = true;
        } else if (isConstRefBoundaryOpaque(newInputs[i], name)) {
          newInputs[i] =
              emitc::OpaqueType::get(&ctxRef, ("const " + name + "&").str());
          entry.getArgument(i).setType(newInputs[i]);
          inputsChanged = true;
        }
      }

      // Pass 2: lift move-only results off the return type. In-place results
      // are dropped entirely (the value already lives in the mutable arg from
      // Pass 1). Other move-only scalars/arrays are re-appended as trailing
      // `T&` / `std::array<T, N>&` out-params, since move-only values and C
      // arrays can't be returned by value in C++.
      enum DpsKind { kInout, kScalar, kArray, kFloatArray };
      SmallVector<size_t> dpsResultIdxs;
      SmallVector<DpsKind> dpsKind;
      SmallVector<Value> dpsOutParam;  // null for kInout
      SmallVector<Type> retainedResults;
      SmallVector<Type> appendedInputs;
      for (auto [i, t] : llvm::enumerate(funcType.getResults())) {
        StringRef name;
        int64_t arraySize = 0;
        if (resultIsInout[i]) {
          dpsResultIdxs.push_back(i);
          dpsKind.push_back(kInout);
          dpsOutParam.push_back(Value{});
          continue;
        }
        if (isMoveOnlyOpaque(t, name)) {
          Type refT = emitc::OpaqueType::get(&ctxRef, (name + "&").str());
          appendedInputs.push_back(refT);
          dpsResultIdxs.push_back(i);
          dpsKind.push_back(kScalar);
          dpsOutParam.push_back(entry.addArgument(refT, op.getLoc()));
          continue;
        }
        if (isMoveOnlyArray(t, name, arraySize)) {
          std::string typeName =
              ("std::array<" + name + ", " + std::to_string(arraySize) + ">&")
                  .str();
          Type refT = emitc::OpaqueType::get(&ctxRef, typeName);
          appendedInputs.push_back(refT);
          dpsResultIdxs.push_back(i);
          dpsKind.push_back(kArray);
          dpsOutParam.push_back(entry.addArgument(refT, op.getLoc()));
          continue;
        }
        // A plain (non-move-only) float array result -- e.g. the decoded
        // message a `*__decrypt__*` client function hands back. A C array
        // can't be returned by value either, so lift it to a trailing `T*`
        // out-param; the return copies the local array's elements into it.
        if (auto arrT = dyn_cast<emitc::ArrayType>(t)) {
          if (isa<FloatType>(arrT.getElementType())) {
            Type ptrT = emitc::PointerType::get(arrT.getElementType());
            appendedInputs.push_back(ptrT);
            dpsResultIdxs.push_back(i);
            dpsKind.push_back(kFloatArray);
            dpsOutParam.push_back(entry.addArgument(ptrT, op.getLoc()));
            continue;
          }
        }
        retainedResults.push_back(t);
      }

      if (!inputsChanged && dpsResultIdxs.empty()) return;

      SmallVector<Type> finalInputs;
      finalInputs.append(newInputs.begin(), newInputs.end());
      finalInputs.append(appendedInputs.begin(), appendedInputs.end());
      op.setType(FunctionType::get(&ctxRef, finalInputs, retainedResults));

      if (dpsResultIdxs.empty()) return;

      // Rewrite each func.return: move scalar/array results into their
      // out-params, drop in-place results (already in the mutable arg), retain
      // the rest.
      SmallVector<emitc::LoadOp> loadsToMaybeErase;
      for (auto ret : returns) {
        OpBuilder b(ret);
        SmallVector<Value> retained;
        size_t cursor = 0;
        for (auto [i, val] : llvm::enumerate(ret.getOperands())) {
          if (cursor < dpsResultIdxs.size() && dpsResultIdxs[cursor] == i) {
            switch (dpsKind[cursor]) {
              case kInout:
                // Value already lives in the mutable in-place argument; the
                // return just drops it.
                break;
              case kArray:
                // Bulk move from the local emitc.array into the std::array
                // out-param via `std::move(begin, end, out.begin())`.
                emitc::VerbatimOp::create(
                    b, ret.getLoc(),
                    "std::move(std::begin({}), std::end({}), {}.begin());",
                    ValueRange{val, val, dpsOutParam[cursor]});
                break;
              case kFloatArray: {
                // Copy the local float array's elements into the `T*`
                // out-param. `&val[0]...[0]` flattens the (row-major) C array
                // to a pointer to its first element.
                auto arrT = cast<emitc::ArrayType>(val.getType());
                int64_t n = 1;
                std::string flat = "&{}";
                for (int64_t d : arrT.getShape()) {
                  n *= d;
                  flat += "[0]";
                }
                emitc::VerbatimOp::create(
                    b, ret.getLoc(),
                    "for (size_t _i = 0; _i < " + std::to_string(n) +
                        "; ++_i) {}[_i] = (" + flat + ")[_i];",
                    ValueRange{dpsOutParam[cursor], val});
                break;
              }
              case kScalar: {
                // Prefer the load's source lvalue (skips a spurious copy);
                // fall back to the value otherwise.
                Value source = val;
                if (auto loadOp = val.getDefiningOp<emitc::LoadOp>()) {
                  source = loadOp.getOperand();
                  loadsToMaybeErase.push_back(loadOp);
                }
                emitc::VerbatimOp::create(
                    b, ret.getLoc(), "{} = std::move({});",
                    ValueRange{dpsOutParam[cursor], source});
                break;
              }
            }
            ++cursor;
          } else {
            retained.push_back(val);
          }
        }
        func::ReturnOp::create(b, ret.getLoc(), retained);
        ret.erase();
      }
      for (auto load : loadsToMaybeErase) {
        if (load.use_empty()) load.erase();
      }
    });

    // Strip leftover `tensor_ext.*` metadata (original-type / layout
    // annotations carried on the client function boundaries). They are
    // HEIR-internal and reference the unregistered `tensor_ext` dialect, which
    // mlir-translate (mlir-to-cpp) cannot parse.
    getOperation()->walk([](func::FuncOp fn) {
      auto stripTensorExt =
          [](DictionaryAttr d) -> std::optional<SmallVector<NamedAttribute>> {
        if (!d) return std::nullopt;
        SmallVector<NamedAttribute> kept;
        for (NamedAttribute a : d)
          if (!a.getName().strref().starts_with("tensor_ext."))
            kept.push_back(a);
        if (kept.size() == d.size()) return std::nullopt;
        return kept;
      };
      for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i)
        if (auto kept = stripTensorExt(fn.getArgAttrDict(i)))
          fn.setArgAttrs(i, *kept);
      for (unsigned i = 0, e = fn.getNumResults(); i < e; ++i)
        if (auto kept = stripTensorExt(fn.getResultAttrDict(i)))
          fn.setResultAttrs(i, *kept);
    });
  }
};

}  // namespace

void registerCheddarToEmitCExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, mlir::emitc::EmitCDialect*) {
    mlir::emitc::OpaqueType::attachInterface<EmitCOpaqueAsMemRefElement>(*ctx);
  });
}

}  // namespace mlir::heir
