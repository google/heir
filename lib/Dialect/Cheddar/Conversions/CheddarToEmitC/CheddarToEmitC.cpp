#include "lib/Dialect/Cheddar/Conversions/CheddarToEmitC/CheddarToEmitC.h"

#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "llvm/include/llvm/ADT/DenseSet.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/StringMap.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringSet.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Utils/MemRefUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CHEDDAREMITCBOUNDARY
#include "lib/Dialect/Cheddar/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

namespace {

using ::mlir::emitc::CallOpaqueOp;
using ::mlir::emitc::LValueType;
using ::mlir::emitc::MemberCallOpaqueOp;
using ::mlir::emitc::OpaqueAttr;
using ::mlir::emitc::OpaqueType;
using ::mlir::emitc::PointerType;
using ::mlir::emitc::VariableOp;
using ::mlir::emitc::VerbatimOp;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

constexpr StringLiteral kDestinationOperandAttr = "cheddar.destination_operand";

template <typename OpTy>
OpTy markDestination(OpTy op, unsigned operandNumber) {
  op->setAttr(
      kDestinationOperandAttr,
      IntegerAttr::get(IntegerType::get(op.getContext(), 64), operandNumber));
  return op;
}

std::optional<unsigned> getDestinationOperand(Operation* op) {
  auto attr = op->getAttrOfType<IntegerAttr>(kDestinationOperandAttr);
  if (!attr) return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

// The CHEDDAR payload C++ type name for a cheddar element type, or "" if `t`
// isn't a (move-only) cheddar payload type.
std::string payloadTypeName(Type t) {
  if (isa<cheddar::CiphertextType>(t)) return "Ciphertext<word>";
  if (isa<cheddar::PlaintextType>(t)) return "Plaintext<word>";
  if (isa<cheddar::ConstantType>(t)) return "Constant<word>";
  if (isa<cheddar::EvalKeyType>(t)) return "EvaluationKey<word>";
  return "";
}

int64_t numElements(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (int64_t dim : shape) result *= dim;
  return result;
}

// A rank >= 1 payload buffer: an `emitc.array` of a CHEDDAR (opaque) type.
bool isPayloadArray(Type t) {
  auto array = dyn_cast<emitc::ArrayType>(t);
  return array && isa<emitc::OpaqueType>(array.getElementType());
}

// The borrowed evaluation key type, as returned by the EvkMap key getters.
Type evalKeyRefType(MLIRContext* ctx) {
  return OpaqueType::get(ctx, "const EvaluationKey<word>&");
}

// `for (size_t i = 0; i < n; i += 1) { body(i) }`. `body` must not create the
// terminator.
void emitForLoop(OpBuilder& b, Location loc, int64_t n,
                 function_ref<void(OpBuilder&, Location, Value)> body) {
  auto sizeT = emitc::SizeTType::get(b.getContext());
  Value lb = emitc::LiteralOp::create(b, loc, sizeT, "0");
  Value ub = emitc::LiteralOp::create(b, loc, sizeT, std::to_string(n));
  Value step = emitc::LiteralOp::create(b, loc, sizeT, "1");
  emitc::ForOp::create(b, loc, lb, ub, step,
                       [&](OpBuilder& b, Location loc, Value iv) {
                         body(b, loc, iv);
                         emitc::YieldOp::create(b, loc);
                       });
}

SmallVector<Value> flattenIndices(OpBuilder& builder, Location loc,
                                  ArrayRef<int64_t> shape, ValueRange indices) {
  auto sizeT = emitc::SizeTType::get(builder.getContext());
  if (indices.empty())
    return {emitc::LiteralOp::create(builder, loc, sizeT, "0")};
  if (indices.size() == 1) return {indices.front()};

  Value flat = indices.front();
  for (size_t i = 1; i < indices.size(); ++i) {
    Value dim =
        emitc::LiteralOp::create(builder, loc, sizeT, std::to_string(shape[i]));
    flat = emitc::MulOp::create(builder, loc, TypeRange{flat.getType()}, flat,
                                dim);
    flat = emitc::AddOp::create(builder, loc, TypeRange{flat.getType()}, flat,
                                indices[i]);
  }
  return {flat};
}

// C++ definitions of the `heir::` helpers the accessor patterns call, emitted
// verbatim into the generated code by --cheddar-emitc-boundary. They isolate
// the accessors whose spelling differs between CHEDDAR-derived APIs, so that
// supporting another API only requires emitting a different version.
// clang-format off
constexpr StringLiteral kCheddarRuntime = R"cpp(namespace heir {
template <typename Context>
const auto& getEncoder(const Context* context) {
  return context->encoder_;
}
template <typename Keys, typename Context>
const auto& multiplicationKey(const Keys& keys, const Context* /*context*/) {
  return keys.GetMultiplicationKey();
}
}  // namespace heir)cpp";
// clang-format on

// Flat `<elt>*` to a buffer's first element: the value itself if it is already
// a pointer, else `&array[0]..[0]`.
Value addressOfFirstElement(OpBuilder& b, Location loc, Value array) {
  if (isa<emitc::PointerType>(array.getType())) return array;
  auto arrayTy = cast<emitc::ArrayType>(array.getType());
  auto sizeT = emitc::SizeTType::get(b.getContext());
  SmallVector<Value> zeroIdxs;
  for (size_t i = 0; i < arrayTy.getShape().size(); ++i)
    zeroIdxs.push_back(emitc::LiteralOp::create(b, loc, sizeT, "0"));
  auto lvalueTy = emitc::LValueType::get(arrayTy.getElementType());
  Value firstElement =
      emitc::SubscriptOp::create(b, loc, lvalueTy, array, zeroIdxs);
  return emitc::AddressOfOp::create(
      b, loc, emitc::PointerType::get(arrayTy.getElementType()), firstElement);
}

// An argument of an opaque call: an SSA operand, or an attribute printed as a
// literal (e.g. an IntegerAttr level or BoolAttr flag).
using CallArg = std::variant<Value, Attribute>;

// Emit `receiver.method(args...)` (or `receiver->method(...)` for a pointer
// receiver).
MemberCallOpaqueOp emitMemberCall(OpBuilder& b, Location loc,
                                  TypeRange resultTypes, Value receiver,
                                  StringRef method, ArrayRef<CallArg> args) {
  SmallVector<Value> operands;
  SmallVector<Attribute> argAttrs;
  bool hasLiteral = false;
  for (const CallArg& arg : args) {
    if (const auto* value = std::get_if<Value>(&arg)) {
      argAttrs.push_back(b.getIndexAttr(operands.size()));
      operands.push_back(*value);
    } else {
      argAttrs.push_back(std::get<Attribute>(arg));
      hasLiteral = true;
    }
  }
  return MemberCallOpaqueOp::create(
      b, loc, resultTypes, receiver, b.getStringAttr(method),
      hasLiteral ? b.getArrayAttr(argAttrs) : ArrayAttr{},
      /*template_args=*/ArrayAttr{}, operands);
}

// Emit `receiver.method(out, args...)`, marking `out` as written.
void emitOutParamCall(OpBuilder& b, Location loc, Value receiver,
                      StringRef method, Value out, ArrayRef<CallArg> args) {
  SmallVector<CallArg> allArgs{out};
  allArgs.append(args.begin(), args.end());
  // Operand 0 is the receiver, so `out` is operand 1.
  markDestination(
      emitMemberCall(b, loc, /*resultTypes=*/{}, receiver, method, allArgs), 1);
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

// Cheddar handle/payload types and payload buffers. A payload buffer
// (`memref<!cheddar.X>`) is an `emitc.lvalue` of the payload type (rank 0) or
// an `emitc.array` of it (rank >= 1); the boundary pass re-types function
// arguments of these types.
void addCheddarEmitCTypeConversions(TypeConverter& tc, MLIRContext* ctx) {
  // Identity for lvalue, which the shared EmitCTypeConverter rejects.
  tc.addConversion([](emitc::LValueType t) -> Type { return t; });
  tc.addConversion([ctx](cheddar::ParameterType) -> Type {
    return OpaqueType::get(ctx, "Parameter<word>");
  });
  tc.addConversion([ctx](cheddar::ContextType) -> Type {
    return PointerType::get(ctx, OpaqueType::get(ctx, "Context<word>"));
  });
  tc.addConversion([ctx](cheddar::BootContextType) -> Type {
    return PointerType::get(ctx, OpaqueType::get(ctx, "BootContext<word>"));
  });
  tc.addConversion([ctx](cheddar::UserInterfaceType) -> Type {
    return PointerType::get(ctx, OpaqueType::get(ctx, "UserInterface<word>"));
  });
  // The non-copyable handles are only ever borrowed: as arguments and as the
  // results of the get_* accessors.
  tc.addConversion([ctx](cheddar::EncoderType) -> Type {
    return OpaqueType::get(ctx, "const Encoder<word>&");
  });
  tc.addConversion([ctx](cheddar::EvkMapType) -> Type {
    return OpaqueType::get(ctx, "const EvkMap<word>&");
  });
  tc.addConversion(
      [ctx](cheddar::EvalKeyType) -> Type { return evalKeyRefType(ctx); });
  tc.addConversion([ctx](cheddar::CiphertextType) -> Type {
    return OpaqueType::get(ctx, "Ciphertext<word>");
  });
  tc.addConversion([ctx](cheddar::PlaintextType) -> Type {
    return OpaqueType::get(ctx, "Plaintext<word>");
  });
  tc.addConversion([ctx](cheddar::ConstantType) -> Type {
    return OpaqueType::get(ctx, "Constant<word>");
  });
  tc.addConversion(
      [ctx](IndexType) -> Type { return emitc::SizeTType::get(ctx); });
  // Payload buffers: the element lvalue (rank 0) or an `emitc.array` of the
  // payload type (static rank >= 1). Primitive buffers: a flat pointer,
  // whatever their storage (alloc, alloca, global, subview).
  tc.addConversion([ctx](MemRefType type) -> std::optional<Type> {
    Type eltType = type.getElementType();
    std::string payloadName = payloadTypeName(eltType);
    bool payload = !payloadName.empty();
    if (!payload && !isa<FloatType, IntegerType>(eltType)) return std::nullopt;
    if (type.getRank() == 0) {
      if (payload)
        return Type(LValueType::get(OpaqueType::get(ctx, payloadName)));
      return Type(emitc::PointerType::get(eltType));
    }
    if (!type.hasStaticShape() || llvm::is_contained(type.getShape(), 0) ||
        !memref::isStaticShapeAndContiguousRowMajor(type))
      return Type();
    if (payload)
      return Type(emitc::ArrayType::get(type.getShape(),
                                        OpaqueType::get(ctx, payloadName)));
    return Type(emitc::PointerType::get(eltType));
  });
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

// Generic destination-passing op -> a single out-parameter method call:
//   receiver->Method(dest, inputs..., extra)
// where `extra` is an optional trailing literal attribute.
template <typename Op>
struct OutParamDpsPattern : public OpConversionPattern<Op> {
  OutParamDpsPattern(const TypeConverter& tc, MLIRContext* ctx,
                     StringRef method,
                     std::function<Attribute(Op)> extra = nullptr)
      : OpConversionPattern<Op>(tc, ctx),
        method(method.str()),
        extra(std::move(extra)) {}

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    unsigned initIdx = dpsOp.getDpsInitOperand(0)->getOperandNumber();
    auto operands = adaptor.getOperands();
    Value receiver = operands[0];
    Value dest = operands[initIdx];
    SmallVector<CallArg> inputs;
    for (unsigned i = 1; i < operands.size(); ++i)
      if (i != initIdx) inputs.push_back(operands[i]);
    if (extra) inputs.push_back(extra(op));
    emitOutParamCall(rewriter, op.getLoc(), receiver, method, dest, inputs);
    rewriter.eraseOp(op);
    return success();
  }

  std::string method;
  std::function<Attribute(Op)> extra;
};

// Support values derived from a context or key, via helpers emitted by
// --cheddar-emitc-boundary (see kCheddarRuntime).
template <typename Op>
struct ConvertRuntimeAccessor : public OpConversionPattern<Op> {
  ConvertRuntimeAccessor(const TypeConverter& tc, MLIRContext* ctx,
                         StringRef callee)
      : OpConversionPattern<Op>(tc, ctx), callee(callee.str()) {}
  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<CallOpaqueOp>(
        op, TypeRange{this->getTypeConverter()->convertType(op.getType())},
        callee, adaptor.getOperands());
    return success();
  }
  std::string callee;
};

struct ConvertGetEvkMap : public OpConversionPattern<cheddar::GetEvkMapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::GetEvkMapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<MemberCallOpaqueOp>(
        op, TypeRange{getTypeConverter()->convertType(op.getType())},
        adaptor.getUi(), rewriter.getStringAttr("GetEvkMap"), ArrayAttr{},
        ArrayAttr{}, ValueRange{});
    return success();
  }
};

// A `std::vector<Complex>` holding a message, as taken and produced by the
// CHEDDAR encoder.
Type complexVectorType(MLIRContext* ctx) {
  return OpaqueType::get(ctx, "std::vector<Complex>");
}

// `encoder.GetScale(level)`: CHEDDAR's canonical scale for the level.
Value emitGetScale(OpBuilder& b, Location loc, Value encoder,
                   IntegerAttr level) {
  return emitMemberCall(b, loc, b.getF64Type(), encoder, "GetScale", {level})
      .getResult(0);
}

// cheddar.encode: copy the message into a std::vector<Complex>, then encode at
// CHEDDAR's canonical scale for the level.
struct ConvertEncode : public OpConversionPattern<cheddar::EncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext* ctx = rewriter.getContext();
    auto messageType = dyn_cast<ShapedType>(op.getMessage().getType());
    if (!messageType || !messageType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "encode requires a static message shape");
    int64_t n = numElements(messageType.getShape());
    // std::vector<Complex>(begin, begin + n)
    Value begin = addressOfFirstElement(rewriter, loc, adaptor.getMessage());
    Value count = emitc::LiteralOp::create(
        rewriter, loc, OpaqueType::get(ctx, "std::ptrdiff_t"),
        std::to_string(n));
    Value end = emitc::AddOp::create(rewriter, loc, TypeRange{begin.getType()},
                                     begin, count);
    Type vecType = complexVectorType(ctx);
    Value vec =
        CallOpaqueOp::create(rewriter, loc, TypeRange{vecType},
                             "std::vector<Complex>", ValueRange{begin, end})
            .getResult(0);
    Value encoder = adaptor.getEncoder();
    // TODO(#2364): Use scale from op once HEIR can do precise scale tracking.
    Value scale = emitGetScale(rewriter, loc, encoder, op.getLevelAttr());
    emitOutParamCall(rewriter, loc, encoder, "Encode", adaptor.getOutput(),
                     {op.getLevelAttr(), scale, vec});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.encode_constant: encode at the same canonical per-level scale.
struct ConvertEncodeConstant
    : public OpConversionPattern<cheddar::EncodeConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value encoder = adaptor.getEncoder();
    Value scale = emitGetScale(rewriter, loc, encoder, op.getLevelAttr());
    emitOutParamCall(rewriter, loc, encoder, "EncodeConstant",
                     adaptor.getOutput(),
                     {op.getLevelAttr(), scale, adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.decode: decode into a temporary complex vector, copy real parts into
// the float destination buffer.
struct ConvertDecode : public OpConversionPattern<cheddar::DecodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::DecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value dst = adaptor.getValue();
    auto memTy = dyn_cast<MemRefType>(op.getValue().getType());
    if (!memTy || !memTy.hasStaticShape() ||
        !isa<FloatType>(memTy.getElementType()))
      return failure();
    Type elementType = memTy.getElementType();
    auto pointerType = dyn_cast<emitc::PointerType>(dst.getType());
    if (!pointerType || pointerType.getPointee() != elementType)
      return failure();
    int64_t n = numElements(memTy.getShape());
    Location loc = op.getLoc();
    MLIRContext* ctx = rewriter.getContext();
    Type vecType = complexVectorType(ctx);
    Value vec = VariableOp::create(rewriter, loc, LValueType::get(vecType),
                                   OpaqueAttr::get(ctx, ""));
    emitOutParamCall(rewriter, loc, adaptor.getEncoder(), "Decode", vec,
                     {adaptor.getPlaintext()});
    // for (...) dst[i] = (&vec)->at(i).real();
    // Reading through a pointer avoids the copy an emitc.load of `vec` would
    // make.
    // TODO: Once upstream's C++ emitter supports emitc.member_call_opaque
    // inside an emitc.expression (it is missing from the operator precedence
    // table in TranslateToCpp.cpp), wrap `load vec` + `at` + `real` in an
    // emitc.expression instead and drop the address_of.
    Value vecPtr = emitc::AddressOfOp::create(
        rewriter, loc, emitc::PointerType::get(vecType), vec);
    emitForLoop(rewriter, loc, n, [&](OpBuilder& b, Location loc, Value i) {
      Value element = emitMemberCall(b, loc, OpaqueType::get(ctx, "Complex"),
                                     vecPtr, "at", {i})
                          .getResult(0);
      Value real =
          emitMemberCall(b, loc, elementType, element, "real", {}).getResult(0);
      Value slot = emitc::SubscriptOp::create(
          b, loc, LValueType::get(elementType), dst, ValueRange{i});
      emitc::AssignOp::create(b, loc, slot, real);
    });
    rewriter.eraseOp(op);
    return success();
  }
};

// `evk.GetRotationKey(distance)`.
Value emitGetRotationKey(OpBuilder& b, Location loc, Value evk,
                         CallArg distance) {
  return emitMemberCall(b, loc, evalKeyRefType(b.getContext()), evk,
                        "GetRotationKey", {distance})
      .getResult(0);
}

// `evk.GetConjugationKey()`.
Value emitGetConjugationKey(OpBuilder& b, Location loc, Value evk) {
  return emitMemberCall(b, loc, evalKeyRefType(b.getContext()), evk,
                        "GetConjugationKey", {})
      .getResult(0);
}

// HRot/HRotAdd/HConj/HConjAdd: look up the rotation/conjugation key on the
// EvkMap operand, then call the Context method with it.
struct ConvertHRot : public OpConversionPattern<cheddar::HRotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HRotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    CallArg distance = adaptor.getDynamicDistance();
    if (auto staticDistance = op.getStaticDistanceAttr())
      distance = staticDistance;
    Value key = emitGetRotationKey(rewriter, loc, adaptor.getEvk(), distance);
    emitOutParamCall(rewriter, loc, adaptor.getCtx(), "HRot",
                     adaptor.getOutput(), {adaptor.getInput(), key, distance});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHRotAdd : public OpConversionPattern<cheddar::HRotAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HRotAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value key = emitGetRotationKey(rewriter, loc, adaptor.getEvk(),
                                   op.getDistanceAttr());
    emitOutParamCall(
        rewriter, loc, adaptor.getCtx(), "HRotAdd", adaptor.getOutput(),
        {adaptor.getInput(), adaptor.getAddend(), key, op.getDistanceAttr()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHConj : public OpConversionPattern<cheddar::HConjOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value key = emitGetConjugationKey(rewriter, loc, adaptor.getEvk());
    emitOutParamCall(rewriter, loc, adaptor.getCtx(), "HConj",
                     adaptor.getOutput(), {adaptor.getInput(), key});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHConjAdd : public OpConversionPattern<cheddar::HConjAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value key = emitGetConjugationKey(rewriter, loc, adaptor.getEvk());
    emitOutParamCall(rewriter, loc, adaptor.getCtx(), "HConjAdd",
                     adaptor.getOutput(),
                     {adaptor.getInput(), adaptor.getAddend(), key});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// memref op patterns (payload + float)
//===----------------------------------------------------------------------===//

// memref.alloc of a payload buffer -> a local variable; the payload owns its
// device memory.
struct ConvertAllocLocal : public OpConversionPattern<mlir::memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    if (!converted ||
        (!isa<emitc::LValueType>(converted) && !isPayloadArray(converted)))
      return failure();
    auto variable = emitc::VariableOp::create(
        rewriter, op.getLoc(), converted,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(op, variable);
    return success();
  }
};

// memref.load through a flat primitive pointer.
struct ConvertLoadPointer : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto pointerType =
        dyn_cast<emitc::PointerType>(adaptor.getMemref().getType());
    if (!pointerType) return failure();

    Type elementType = op.getMemRefType().getElementType();
    if (pointerType.getPointee() != elementType ||
        !isa<FloatType, IntegerType>(elementType))
      return failure();

    SmallVector<Value> indices =
        flattenIndices(rewriter, op.getLoc(), op.getMemRefType().getShape(),
                       adaptor.getIndices());
    auto subscript = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), emitc::LValueType::get(elementType),
        adaptor.getMemref(), indices);
    auto loaded = emitc::LoadOp::create(rewriter, op.getLoc(), elementType,
                                        subscript.getResult());
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

// memref.dealloc of a payload -> `v = T();`, releasing the device buffer at
// last use instead of at scope exit (peak memory would otherwise be the sum of
// all intermediates). `v = {}` does not compile: the constructors are explicit.
// `T()` is an emitc.literal, which is always inlined, so the assignment is a
// move from a temporary rather than a copy of a named (move-only) value.
struct EraseDealloc : public OpConversionPattern<mlir::memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::DeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value memref = adaptor.getMemref();
    Type memTy = memref.getType();
    auto emitReset = [](OpBuilder& b, Location loc, Value lvalue) {
      auto type =
          cast<OpaqueType>(cast<LValueType>(lvalue.getType()).getValueType());
      Value fresh =
          emitc::LiteralOp::create(b, loc, type, type.getValue().str() + "()");
      emitc::AssignOp::create(b, loc, lvalue, fresh);
    };
    if (isPayloadArray(memTy)) {
      auto array = cast<emitc::ArrayType>(memTy);
      if (array.getRank() == 1)
        emitForLoop(rewriter, loc, array.getShape()[0],
                    [&](OpBuilder& b, Location loc, Value i) {
                      Value slot = emitc::SubscriptOp::create(
                          b, loc, LValueType::get(array.getElementType()),
                          memref, ValueRange{i});
                      emitReset(b, loc, slot);
                    });
      rewriter.eraseOp(op);
      return success();
    }
    if (auto l = dyn_cast<emitc::LValueType>(memTy)) {
      if (isa<emitc::OpaqueType>(l.getValueType()))
        emitReset(rewriter, loc, memref);
      // Other lvalues are scope-bound values with nothing to free.
      rewriter.eraseOp(op);
      return success();
    }
    // A primitive heap buffer.
    if (isa<emitc::PointerType>(memTy)) {
      emitc::CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{}, "free",
                                  ValueRange{adaptor.getMemref()});
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// memref.load on a payload buffer -> `base[i...]`, kept as an lvalue (payloads
// are move-only).
struct ConvertLoadArray : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type baseTy = adaptor.getMemref().getType();
    if (!isa<emitc::LValueType>(baseTy) && !isPayloadArray(baseTy))
      return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    // A rank-0 payload memref converts to the element lvalue itself.
    if (adaptor.getIndices().empty()) {
      if (isa<emitc::OpaqueType>(elt)) {
        rewriter.replaceOp(op, adaptor.getMemref());
        return success();
      }
      return failure();
    }
    auto sub = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), emitc::LValueType::get(elt), adaptor.getMemref(),
        adaptor.getIndices());
    if (isa<emitc::OpaqueType>(elt)) {
      rewriter.replaceOp(op, sub.getResult());  // payload: lvalue, no copy
      return success();
    }
    auto loaded =
        emitc::LoadOp::create(rewriter, op.getLoc(), elt, sub.getResult());
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

// Look through the materialization cast the driver inserts between an
// `lvalue<opaque T>` producer and an `opaque T` use.
static Value unwrapSingleUnrealizedCast(Value v) {
  if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      cast && cast.getInputs().size() == 1 &&
      isa<emitc::LValueType>(cast.getInputs()[0].getType()))
    return cast.getInputs()[0];
  return v;
}

// Copying a payload buffer needs CHEDDAR's deep-copy API, which is not lowered
// yet. Reject it rather than letting the stock MemRefToEmitC pattern emit a
// memcpy of non-trivially-copyable objects. A self-copy is a no-op.
struct RejectPayloadCopy : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    if (payloadTypeName(sourceType.getElementType()).empty()) return failure();
    if (op.getSource() == op.getTarget()) {
      rewriter.eraseOp(op);
      return success();
    }
    return op.emitOpError("copying a Cheddar payload buffer is not supported");
  }
};

// memref.store: assign through the flat pointer (primitive) or into the
// element lvalue (payload). Payload stores stay `emitc.verbatim`: the stored
// value is itself an lvalue, and `emitc.assign` would require an `emitc.load`
// of it first, and the boundary pass later re-types argument lvalues to C++
// references, which `emitc.assign` does not accept.
struct ConvertStoreArray : public OpConversionPattern<mlir::memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type baseTy = adaptor.getMemref().getType();
    bool isPointer = isa<emitc::PointerType>(baseTy);
    if (!isPointer && !isa<emitc::LValueType>(baseTy) &&
        !isPayloadArray(baseTy))
      return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    // Rank-0: the payload lvalue itself; a primitive pointer still subscripts.
    if (!isPointer && adaptor.getIndices().empty()) {
      if (isa<emitc::LValueType>(baseTy) && isa<emitc::OpaqueType>(elt)) {
        markDestination(
            VerbatimOp::create(
                rewriter, op.getLoc(), "{} = {};",
                ValueRange{adaptor.getMemref(),
                           unwrapSingleUnrealizedCast(adaptor.getValue())}),
            0);
        rewriter.eraseOp(op);
        return success();
      }
      return failure();
    }
    SmallVector<Value> indices =
        isPointer ? flattenIndices(rewriter, op.getLoc(),
                                   op.getMemRefType().getShape(),
                                   adaptor.getIndices())
                  : SmallVector<Value>(adaptor.getIndices().begin(),
                                       adaptor.getIndices().end());
    auto sub = emitc::SubscriptOp::create(rewriter, op.getLoc(),
                                          emitc::LValueType::get(elt),
                                          adaptor.getMemref(), indices);
    if (isa<emitc::OpaqueType>(elt)) {
      markDestination(
          VerbatimOp::create(
              rewriter, op.getLoc(), "{} = {};",
              ValueRange{sub.getResult(),
                         unwrapSingleUnrealizedCast(adaptor.getValue())}),
          0);
    } else {
      emitc::AssignOp::create(rewriter, op.getLoc(), sub.getResult(),
                              adaptor.getValue());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.subview selecting one element of a payload array -> `base[o...]`.
struct ConvertSubViewSubscript
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    if (!isPayloadArray(base.getType())) return failure();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!isa_and_present<emitc::LValueType>(resultTy))
      return rewriter.notifyMatchFailure(
          op, "payload subview must select a single element");
    for (int64_t stride : op.getStaticStrides())
      if (stride != 1)
        return rewriter.notifyMatchFailure(op, "expected unit strides");
    auto sizeT = emitc::SizeTType::get(getContext());
    SmallVector<Value> idx;
    unsigned dynCursor = 0;
    for (OpFoldResult offset : op.getMixedOffsets()) {
      if (isa<Value>(offset)) {
        idx.push_back(adaptor.getOffsets()[dynCursor++]);
      } else {
        int64_t o = cast<IntegerAttr>(cast<Attribute>(offset)).getInt();
        idx.push_back(emitc::LiteralOp::create(rewriter, op.getLoc(), sizeT,
                                               std::to_string(o)));
      }
    }
    rewriter.replaceOp(op, emitc::SubscriptOp::create(rewriter, op.getLoc(),
                                                      resultTy, base, idx));
    return success();
  }
};

// memref.cast between layouts of the same payload buffer is a no-op in C++.
struct ConvertPayloadCast : public OpConversionPattern<mlir::memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resTy = dyn_cast<MemRefType>(op.getType());
    if (!resTy || !isa<cheddar::CiphertextType, cheddar::PlaintextType,
                       cheddar::ConstantType>(resTy.getElementType()))
      return failure();
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

// memref.subview producing a contiguous cleartext slice -> pointer arithmetic
// using the source memref's actual static strides.
struct ConvertSubViewToPointer
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    auto pointerTy = dyn_cast<emitc::PointerType>(base.getType());
    Type elementType = pointerTy ? pointerTy.getPointee() : Type{};
    if (!isa_and_present<FloatType, IntegerType>(elementType)) return failure();
    auto resultType = cast<MemRefType>(op.getType());
    if (!memref::isStaticShapeAndContiguousRowMajor(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "float subview must be contiguous");
    auto offsets = op.getStaticOffsets();
    auto sourceType = op.getSourceType();
    if (static_cast<int64_t>(offsets.size()) != sourceType.getRank())
      return failure();
    for (int64_t o : offsets)
      if (ShapedType::isDynamic(o)) return failure();
    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;
    if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)) ||
        sourceOffset != 0 ||
        llvm::is_contained(sourceStrides, ShapedType::kDynamic))
      return failure();
    int64_t linearOffset = 0;
    for (int64_t i = 0; i < sourceType.getRank(); ++i)
      linearOffset += offsets[i] * sourceStrides[i];
    if (linearOffset == 0) {
      rewriter.replaceOp(op, base);
      return success();
    }
    auto literal = emitc::LiteralOp::create(
        rewriter, op.getLoc(),
        emitc::OpaqueType::get(getContext(), "std::ptrdiff_t"),
        std::to_string(linearOffset));
    auto offset = emitc::AddOp::create(
        rewriter, op.getLoc(), TypeRange{base.getType()}, base, literal);
    rewriter.replaceOp(op, offset.getResult());
    return success();
  }
};

// memref.copy between flat primitive pointers -> an element-wise loop.
struct ConvertMemRefCopyPrimitive
    : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value src = adaptor.getSource();
    Value tgt = adaptor.getTarget();
    auto primitiveOperand = [](Value v) -> bool {
      if (auto p = dyn_cast<emitc::PointerType>(v.getType()))
        return isa<FloatType, IntegerType>(p.getPointee());
      return false;
    };
    if (!primitiveOperand(src) || !primitiveOperand(tgt) ||
        src.getType() != tgt.getType())
      return failure();
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    if (!sourceType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "primitive copy requires static shape");
    int64_t n = numElements(sourceType.getShape());
    Type elementType = cast<emitc::PointerType>(src.getType()).getPointee();
    emitForLoop(
        rewriter, op.getLoc(), n, [&](OpBuilder& b, Location loc, Value i) {
          auto lvalueType = LValueType::get(elementType);
          Value from = emitc::SubscriptOp::create(b, loc, lvalueType, src,
                                                  ValueRange{i});
          Value value = emitc::LoadOp::create(b, loc, elementType, from);
          Value to = emitc::SubscriptOp::create(b, loc, lvalueType, tgt,
                                                ValueRange{i});
          emitc::AssignOp::create(b, loc, to, value);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

// Upstream ConvertGlobal, minus its rejection of the `alignment` attribute that
// bufferized constants carry; the global is C-array storage behind the flat
// pointer handle.
struct ConvertGlobalDropAlign
    : public OpConversionPattern<mlir::memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MemRefType type = op.getType();
    if (!type.hasStaticShape() ||
        !isa<FloatType, IntegerType>(type.getElementType()))
      return failure();
    Type storageType = type.getRank() == 0
                           ? type.getElementType()
                           : Type(emitc::ArrayType::get(type.getShape(),
                                                        type.getElementType()));
    auto vis = SymbolTable::getSymbolVisibility(op);
    if (vis != SymbolTable::Visibility::Public &&
        vis != SymbolTable::Visibility::Private)
      return failure();
    bool staticSpecifier = vis == SymbolTable::Visibility::Private;
    Attribute initialValue = adaptor.getInitialValueAttr();
    if (type.getRank() == 0) {
      if (!op.getInitialValue()) return failure();
      auto elements = dyn_cast<ElementsAttr>(*op.getInitialValue());
      if (!elements) return failure();
      initialValue = elements.getSplatValue<Attribute>();
    }
    if (isa_and_present<UnitAttr>(initialValue)) initialValue = {};
    // Non-const: the memref handle is a non-const pointer.
    auto global = emitc::GlobalOp::create(
        rewriter, op.getLoc(), adaptor.getSymName(),
        /*sym_visibility=*/StringAttr{}, storageType, initialValue,
        /*externSpecifier=*/!staticSpecifier, staticSpecifier,
        /*constSpecifier=*/false);
    rewriter.replaceOp(op, global);
    return success();
  }
};

// memref.get_global -> pointer to the global's first element.
struct ConvertGetGlobalPointer
    : public OpConversionPattern<mlir::memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GetGlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MemRefType type = op.getType();
    if (!type.hasStaticShape() ||
        !isa<FloatType, IntegerType>(type.getElementType()))
      return failure();
    auto pointerType = dyn_cast_if_present<emitc::PointerType>(
        getTypeConverter()->convertType(type));
    if (!pointerType || pointerType.getPointee() != type.getElementType())
      return failure();
    if (type.getRank() == 0) {
      auto lvalueType = emitc::LValueType::get(type.getElementType());
      Value global = emitc::GetGlobalOp::create(
          rewriter, op.getLoc(), lvalueType, adaptor.getNameAttr());
      rewriter.replaceOp(op, emitc::AddressOfOp::create(rewriter, op.getLoc(),
                                                        pointerType, global));
      return success();
    }
    auto arrayType =
        emitc::ArrayType::get(type.getShape(), type.getElementType());
    Value global = emitc::GetGlobalOp::create(rewriter, op.getLoc(), arrayType,
                                              adaptor.getNameAttr());
    rewriter.replaceOp(op,
                       addressOfFirstElement(rewriter, op.getLoc(), global));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertToEmitC dialect interface
//===----------------------------------------------------------------------===//

// Legality, type conversions and patterns for `--convert-to-emitc`.
struct CheddarToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  CheddarToEmitCDialectInterface(Dialect* dialect)
      : ConvertToEmitCPatternInterface(dialect) {}

  void populateConvertToEmitCConversionPatterns(
      ConversionTarget& target, TypeConverter& typeConverter,
      RewritePatternSet& patterns,
      std::optional<bool> /*lowerToCpp*/) const final {
    MLIRContext* ctx = patterns.getContext();
    addCheddarEmitCTypeConversions(typeConverter, ctx);

    // Keep func.func; convert only its signature (checking body legality here
    // would be circular).
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&typeConverter](func::ReturnOp op) {
          return typeConverter.isLegal(op);
        });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&typeConverter](func::CallOp op) {
          return typeConverter.isLegal(op);
        });

    target.addIllegalDialect<cheddar::CheddarDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addDynamicallyLegalDialect<mlir::memref::MemRefDialect>(
        [&typeConverter](Operation* op) { return typeConverter.isLegal(op); });
    // memref.global has no operands/results, so isLegal() would always pass.
    target.addIllegalOp<mlir::memref::GlobalOp>();

    // Stock MemRefToEmitC patterns at default benefit; ours below win at 2.
    mlir::populateMemRefToEmitCConversionPatterns(patterns, typeConverter);

    patterns.add<ConvertAllocLocal, EraseDealloc, ConvertLoadPointer,
                 ConvertLoadArray, ConvertStoreArray,
                 ConvertMemRefCopyPrimitive, ConvertSubViewSubscript,
                 ConvertSubViewToPointer, ConvertPayloadCast,
                 ConvertGlobalDropAlign, ConvertGetGlobalPointer>(
        typeConverter, ctx, /*benefit=*/2);
    patterns.add<RejectPayloadCopy>(typeConverter, ctx, /*benefit=*/3);

    patterns
        .add<ConvertEncode, ConvertEncodeConstant, ConvertDecode, ConvertHRot,
             ConvertHRotAdd, ConvertHConj, ConvertHConjAdd, ConvertGetEvkMap>(
            typeConverter, ctx);
    patterns.add<ConvertRuntimeAccessor<cheddar::GetEncoderOp>>(
        typeConverter, ctx, "heir::getEncoder");
    patterns.add<ConvertRuntimeAccessor<cheddar::GetMultKeyOp>>(
        typeConverter, ctx, "heir::multiplicationKey");

    auto addDps = [&](StringRef name, auto opTag,
                      std::function<Attribute(decltype(opTag))> extra =
                          nullptr) {
      using Op = decltype(opTag);
      patterns.add<OutParamDpsPattern<Op>>(typeConverter, ctx, name, extra);
    };
    addDps("Add", cheddar::AddOp{});
    addDps("Sub", cheddar::SubOp{});
    addDps("Mult", cheddar::MultOp{});
    addDps("Add", cheddar::AddPlainOp{});
    addDps("Sub", cheddar::SubPlainOp{});
    addDps("Mult", cheddar::MultPlainOp{});
    addDps("Add", cheddar::AddConstOp{});
    addDps("Mult", cheddar::MultConstOp{});
    addDps("Neg", cheddar::NegOp{});
    addDps("Rescale", cheddar::RescaleOp{});
    addDps("Relinearize", cheddar::RelinearizeOp{});
    addDps("RelinearizeRescale", cheddar::RelinearizeRescaleOp{});
    addDps("Encrypt", cheddar::EncryptOp{});
    addDps("Decrypt", cheddar::DecryptOp{});
    addDps("MadUnsafe", cheddar::MadUnsafeOp{});
    addDps("Boot", cheddar::BootOp{});
    addDps("LevelDown", cheddar::LevelDownOp{},
           [](cheddar::LevelDownOp op) -> Attribute {
             return op.getTargetLevelAttr();
           });
    addDps("HMult", cheddar::HMultOp{}, [](cheddar::HMultOp op) -> Attribute {
      return BoolAttr::get(op.getContext(), op.getRescale());
    });
  }
};

//===----------------------------------------------------------------------===//
// cheddar-emitc-boundary pass
//===----------------------------------------------------------------------===//

// Scalar payload argument: `T&` (written) / `const T&` (read-only). Payload
// array argument: stays an `emitc.array` of `T` (printed `T name[N]`) when
// written, becomes an `emitc.array` of `const T` when read-only.
Type referenceArgType(MLIRContext* ctx, Type converted, bool written) {
  if (isPayloadArray(converted)) {
    if (written) return {};
    auto array = cast<emitc::ArrayType>(converted);
    return emitc::ArrayType::get(
        array.getShape(),
        OpaqueType::get(
            ctx,
            "const " +
                cast<OpaqueType>(array.getElementType()).getValue().str()));
  }
  if (auto l = dyn_cast<emitc::LValueType>(converted)) {
    auto o = dyn_cast<emitc::OpaqueType>(l.getValueType());
    if (!o) return {};
    std::string base = o.getValue().str();
    return OpaqueType::get(ctx,
                           written ? (base + "&") : ("const " + base + "&"));
  }
  return {};
}

// Is `root` (or a subscript/cast of it) a marked destination, or passed to a
// written callee argument?
bool valueWrittenAsDest(
    Value root,
    const llvm::StringMap<SmallVector<bool>>& writtenFunctionArguments) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) continue;
    for (OpOperand& use : value.getUses()) {
      Operation* owner = use.getOwner();
      unsigned operandNumber = use.getOperandNumber();
      if (getDestinationOperand(owner) == operandNumber) return true;
      if (auto call = dyn_cast<func::CallOp>(owner)) {
        auto it = writtenFunctionArguments.find(call.getCallee());
        if (it != writtenFunctionArguments.end() &&
            operandNumber < it->second.size() && it->second[operandNumber])
          return true;
      }
      if (auto subscript = dyn_cast<emitc::SubscriptOp>(owner);
          subscript && operandNumber == 0)
        worklist.push_back(subscript.getResult());
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(owner);
          cast && cast->getNumResults() == 1)
        worklist.push_back(cast->getResult(0));
    }
  }
  return false;
}

struct CheddarEmitCBoundary
    : public impl::CheddarEmitCBoundaryBase<CheddarEmitCBoundary> {
  using CheddarEmitCBoundaryBase::CheddarEmitCBoundaryBase;

  void runOnOperation() override {
    auto* ctx = &getContext();
    ModuleOp module = getOperation();

    // The runtime helper definitions to emit alongside the generated code.
    StringRef runtimeSource = kCheddarRuntime;

    // Headers and declarations the emitted C++ relies on, so the translated
    // output compiles as-is against the CHEDDAR library.
    OpBuilder b(ctx);
    b.setInsertionPointToStart(module.getBody());
    for (StringRef header : {"array", "cmath", "complex", "cstdint", "cstdlib",
                             "fstream", "memory", "utility", "vector"})
      emitc::IncludeOp::create(b, module.getLoc(), header,
                               /*isStandardInclude=*/true);
    for (StringRef header :
         {"UserInterface.h", "core/Context.h", "core/Encode.h",
          "core/Parameter.h", "extension/BootContext.h"})
      emitc::IncludeOp::create(b, module.getLoc(), header,
                               /*isStandardInclude=*/false);
    VerbatimOp::create(b, module.getLoc(), "using namespace cheddar;");
    VerbatimOp::create(b, module.getLoc(), "using word = uint64_t;");
    VerbatimOp::create(b, module.getLoc(), runtimeSource);

    // Written arguments: seeded from `bufferize.result`, closed over call
    // edges.
    llvm::StringMap<SmallVector<bool>> writtenFunctionArguments;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal()) return;
      SmallVector<bool> written(fn.getNumArguments(), false);
      for (unsigned i = 0; i < fn.getNumArguments(); ++i)
        written[i] = static_cast<bool>(fn.getArgAttr(i, "bufferize.result"));
      writtenFunctionArguments[fn.getName()] = std::move(written);
    });
    bool changedWritten;
    do {
      changedWritten = false;
      getOperation()->walk([&](func::FuncOp fn) {
        if (fn.isExternal()) return;
        SmallVector<bool>& written = writtenFunctionArguments[fn.getName()];
        for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
          if (written[i]) continue;
          if (valueWrittenAsDest(fn.getArgument(i), writtenFunctionArguments)) {
            written[i] = true;
            changedWritten = true;
          }
        }
      });
    } while (changedWritten);

    // Payload lvalue arguments -> C++ references, mutable iff written.
    llvm::StringSet<> refified;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal()) return;
      Block& entry = fn.getBody().front();
      SmallVector<Type> inputs(fn.getFunctionType().getInputs().begin(),
                               fn.getFunctionType().getInputs().end());
      bool changed = false;
      for (unsigned i = 0; i < inputs.size(); ++i) {
        bool written = writtenFunctionArguments[fn.getName()][i];
        Type ref = referenceArgType(ctx, inputs[i], written);
        if (!ref) continue;
        inputs[i] = ref;
        entry.getArgument(i).setType(ref);
        // Subscripts of a re-typed array argument yield its (const) element.
        if (auto array = dyn_cast<emitc::ArrayType>(ref))
          for (Operation* user : entry.getArgument(i).getUsers())
            if (auto subscript = dyn_cast<emitc::SubscriptOp>(user))
              subscript.getResult().setType(
                  emitc::LValueType::get(array.getElementType()));
        changed = true;
      }
      if (changed) {
        fn.setType(
            FunctionType::get(ctx, inputs, fn.getFunctionType().getResults()));
        refified.insert(fn.getName());
      }
    });

    // Calls to re-typed callees no longer type-check as func.call.
    SmallVector<func::CallOp> callsToRewrite;
    getOperation()->walk([&](func::CallOp call) {
      if (refified.contains(call.getCallee())) callsToRewrite.push_back(call);
    });
    for (func::CallOp call : callsToRewrite) {
      OpBuilder b(call);
      auto rewritten = CallOpaqueOp::create(
          b, call.getLoc(), call.getResultTypes(),
          b.getStringAttr(call.getCallee()), /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{}, call.getOperands());
      call.replaceAllUsesWith(rewritten.getResults());
      call.erase();
    }

    getOperation()->walk(
        [](Operation* op) { op->removeAttr(kDestinationOperandAttr); });
  }
};

}  // namespace

void registerCheddarConvertToEmitCInterface(DialectRegistry& registry) {
  registry.addExtension(
      +[](MLIRContext* ctx, cheddar::CheddarDialect* dialect) {
        dialect->addInterfaces<CheddarToEmitCDialectInterface>();
      });
}

}  // namespace mlir::heir
