#include "lib/Transforms/PropagatePadding/PropagatePadding.h"

#include <cstddef>
#include <cstdint>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "propagate-padding"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PROPAGATEPADDING
#include "lib/Transforms/PropagatePadding/PropagatePadding.h.inc"

using tensor_ext::PaddingAttr;

namespace {

PaddingAttr getPadding(Attribute attr) {
  return dyn_cast_or_null<PaddingAttr>(attr);
}

bool isZeroAttr(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

DenseElementsAttr getAsDenseElementsAttr(Value value) {
  ElementsAttr elements;
  if (!matchPattern(value, m_Constant(&elements))) return nullptr;
  if (auto dense = dyn_cast<DenseElementsAttr>(elements)) return dense;
  if (auto res = dyn_cast<DenseResourceElementsAttr>(elements)) {
    return DenseElementsAttr::getFromRawBuffer(res.getType(), res.getData());
  }
  return nullptr;
}

// Whether `value` is statically known to be zero everywhere: a splat-zero
// constant or a linalg.fill of a zero scalar.
bool isKnownAllZero(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    Value scalar = fill.getInputs()[0];
    return matchPattern(scalar, m_AnyZeroFloat()) ||
           matchPattern(scalar, m_Zero());
  }
  if (DenseElementsAttr dense = getAsDenseElementsAttr(value))
    return dense.isSplat() && isZeroAttr(dense.getSplatValue<Attribute>());
  return false;
}

// Whether `value` is a static dense constant that is zero at every index
// outside the trailing-pad logical region `logical`.
bool constantPadRegionIsZero(Value value, ArrayRef<int64_t> logical) {
  if (auto addOp = value.getDefiningOp<arith::AddFOp>()) {
    return constantPadRegionIsZero(addOp.getLhs(), logical) &&
           constantPadRegionIsZero(addOp.getRhs(), logical);
  }
  DenseElementsAttr dense = getAsDenseElementsAttr(value);
  if (!dense) return false;
  auto type = dyn_cast<RankedTensorType>(dense.getType());
  if (!type || !type.hasStaticShape()) return false;
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() != logical.size()) return false;
  if (dense.isSplat())
    return shape == logical || isZeroAttr(dense.getSplatValue<Attribute>());

  SmallVector<int64_t> idx(shape.size(), 0);
  for (Attribute element : dense.getValues<Attribute>()) {
    bool inPad = false;
    for (size_t d = 0; d < shape.size(); ++d) {
      if (idx[d] >= logical[d]) {
        inPad = true;
        break;
      }
    }
    if (inPad && !isZeroAttr(element)) return false;
    for (int64_t d = shape.size() - 1; d >= 0; --d) {
      if (++idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
  return true;
}

// ----------------------------------------------------------------------
// External models
// ----------------------------------------------------------------------

// Seed: tensor.pad with static shapes, no low padding, and a constant
// padding value. A zero padding value yields zero_padded = true; any other
// constant still yields valid shape information with zero_padded = false.
struct PadOpPaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<PadOpPaddingImpl,
                                                        tensor::PadOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<tensor::PadOp>(opAbs);
    if (llvm::any_of(op.getStaticLow(), [](int64_t v) { return v != 0; }))
      return nullptr;
    if (llvm::any_of(op.getStaticHigh(),
                     [](int64_t v) { return ShapedType::isDynamic(v); }))
      return nullptr;
    Value padValue = op.getConstantPaddingValue();
    if (!padValue) return nullptr;
    auto srcType = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!srcType || !resType || !srcType.hasStaticShape() ||
        !resType.hasStaticShape())
      return nullptr;
    bool zero = matchPattern(padValue, m_AnyZeroFloat()) ||
                matchPattern(padValue, m_Zero());

    PaddingAttr inputPadding = getPadding(operandPaddings[0]);
    ArrayRef<int64_t> logical =
        inputPadding ? inputPadding.logical() : srcType.getShape();
    if (inputPadding && !inputPadding.isZeroPadded()) zero = false;

    return PaddingAttr::get(op.getContext(), logical, resType.getShape(), zero);
  }
};

// linalg.matmul / linalg.batch_matmul. Requires both inputs zero-padded (a
// nonzero entry in either operand's padding region would pollute every
// contraction it participates in) and an all-zero or compatibly-padded
// accumulator init.
template <typename OpTy>
struct MatmulPaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<MatmulPaddingImpl<OpTy>,
                                                        OpTy> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<OpTy>(opAbs);
    PaddingAttr lhs = getPadding(operandPaddings[0]);
    PaddingAttr rhs = getPadding(operandPaddings[1]);
    if (!lhs || !rhs) return nullptr;
    if (!lhs.isZeroPadded() || !rhs.isZeroPadded()) return nullptr;

    ArrayRef<int64_t> lhsL = lhs.logical();
    ArrayRef<int64_t> rhsL = rhs.logical();
    size_t rank = lhsL.size();
    bool batch = rank == 3;
    if (rhsL.size() != rank || (rank != 2 && rank != 3)) return nullptr;
    // Batch logical dims must agree.
    if (batch && lhsL[0] != rhsL[0]) return nullptr;

    SmallVector<int64_t> logical;
    SmallVector<int64_t> padded;
    if (batch) {
      logical = {lhsL[0], lhsL[1], rhsL[2]};
      padded = {lhs.padded()[0], lhs.padded()[1], rhs.padded()[2]};
    } else {
      logical = {lhsL[0], rhsL[1]};
      padded = {lhs.padded()[0], rhs.padded()[1]};
    }

    // The accumulator is added to the product, so it must be all-zero or
    // itself compatibly zero-padded.
    Value init = op.getOutputs()[0];
    PaddingAttr initPadding = getPadding(operandPaddings[2]);
    bool initOk = isKnownAllZero(init) ||
                  (initPadding && initPadding.isZeroPadded() &&
                   initPadding.logical() == ArrayRef<int64_t>(logical) &&
                   initPadding.padded() == ArrayRef<int64_t>(padded)) ||
                  constantPadRegionIsZero(init, logical);
    if (!initOk) return nullptr;

    return PaddingAttr::get(op.getContext(), logical, padded, true);
  }
};

// Elementwise binary arith ops on identical shapes. `ZeroAbsorbing` marks
// multiplication: a zero-padded side forces zero pads in the result when the
// other side is a (finite) constant.
template <typename OpTy, bool ZeroAbsorbing>
struct BinaryElementwisePaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<
          BinaryElementwisePaddingImpl<OpTy, ZeroAbsorbing>, OpTy> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<OpTy>(opAbs);
    PaddingAttr lhs = getPadding(operandPaddings[0]);
    PaddingAttr rhs = getPadding(operandPaddings[1]);
    MLIRContext* ctx = op.getContext();

    if (lhs && rhs) {
      if (lhs.logical() != rhs.logical() || lhs.padded() != rhs.padded())
        return nullptr;
      bool zero = lhs.isZeroPadded() && rhs.isZeroPadded();
      return PaddingAttr::get(ctx, lhs.logical(), lhs.padded(), zero);
    }

    // One side carries state; the other must be a constant to reason about.
    PaddingAttr padding = lhs ? lhs : rhs;
    if (!padding) return nullptr;
    Value other = op->getOperand(lhs ? 1 : 0);
    if (!getAsDenseElementsAttr(other)) return nullptr;

    bool zero;
    if (ZeroAbsorbing) {
      // mul: 0 * c = 0 for any finite constant c.
      zero = padding.isZeroPadded();
    } else {
      // add/sub: pads stay zero only if both sides' pads are zero.
      zero = padding.isZeroPadded() &&
             constantPadRegionIsZero(other, padding.logical());
    }
    return PaddingAttr::get(ctx, padding.logical(), padding.padded(), zero);
  }
};

// Elementwise unary ops: shape info always transfers; the zero flag survives
// only for zero-preserving functions (f(0) = 0).
template <typename OpTy, bool ZeroPreserving>
struct UnaryElementwisePaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<
          UnaryElementwisePaddingImpl<OpTy, ZeroPreserving>, OpTy> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    PaddingAttr padding = getPadding(operandPaddings[0]);
    if (!padding) return nullptr;
    bool zero = ZeroPreserving && padding.isZeroPadded();
    return PaddingAttr::get(opAbs->getContext(), padding.logical(),
                            padding.padded(), zero);
  }
};

struct TransposePaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<TransposePaddingImpl,
                                                        linalg::TransposeOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<linalg::TransposeOp>(opAbs);
    PaddingAttr padding = getPadding(operandPaddings[0]);
    if (!padding) return nullptr;
    ArrayRef<int64_t> perm = op.getPermutation();
    SmallVector<int64_t> logical, padded;
    for (int64_t d : perm) {
      logical.push_back(padding.logical()[d]);
      padded.push_back(padding.padded()[d]);
    }
    return PaddingAttr::get(op.getContext(), logical, padded,
                            padding.isZeroPadded());
  }
};

struct BroadcastPaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<BroadcastPaddingImpl,
                                                        linalg::BroadcastOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<linalg::BroadcastOp>(opAbs);
    PaddingAttr padding = getPadding(operandPaddings[0]);
    if (!padding) return nullptr;
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) return nullptr;
    ArrayRef<int64_t> resultShape = resultType.getShape();
    DenseSet<int64_t> added(op.getDimensions().begin(),
                            op.getDimensions().end());
    SmallVector<int64_t> logical, padded;
    size_t srcDim = 0;
    for (int64_t d = 0; d < resultType.getRank(); ++d) {
      if (added.contains(d)) {
        // Broadcast copies are all valid data: logical == padded.
        logical.push_back(resultShape[d]);
        padded.push_back(resultShape[d]);
      } else {
        logical.push_back(padding.logical()[srcDim]);
        padded.push_back(padding.padded()[srcDim]);
        srcDim++;
      }
    }
    return PaddingAttr::get(op.getContext(), logical, padded,
                            padding.isZeroPadded());
  }
};

// linalg.reduce with a single add-reduction: the reduced dimensions must be
// zero-padded (their pad entries are summed INTO the logical results) and
// the init must be all-zero.
struct ReducePaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<ReducePaddingImpl,
                                                        linalg::ReduceOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<linalg::ReduceOp>(opAbs);
    if (op.getInputs().size() != 1 || op.getInits().size() != 1) return nullptr;
    PaddingAttr padding = getPadding(operandPaddings[0]);
    if (!padding) return nullptr;
    // Body must be a single addf/addi + yield.
    Block& body = op.getRegion().front();
    if (body.getOperations().size() != 2) return nullptr;
    Operation& combiner = body.front();
    if (!isa<arith::AddFOp, arith::AddIOp>(combiner)) return nullptr;
    if (!isKnownAllZero(op.getInits()[0])) return nullptr;

    DenseSet<int64_t> reduced(op.getDimensions().begin(),
                              op.getDimensions().end());
    // Pad entries along reduced dims are summed into logical results.
    for (int64_t d : reduced) {
      if (padding.logical()[d] != padding.padded()[d] &&
          !padding.isZeroPadded())
        return nullptr;
    }
    SmallVector<int64_t> logical, padded;
    for (size_t d = 0; d < padding.logical().size(); ++d) {
      if (reduced.contains(d)) continue;
      logical.push_back(padding.logical()[d]);
      padded.push_back(padding.padded()[d]);
    }
    return PaddingAttr::get(op.getContext(), logical, padded,
                            padding.isZeroPadded());
  }
};

struct ExpandShapePaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<ExpandShapePaddingImpl,
                                                        tensor::ExpandShapeOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<tensor::ExpandShapeOp>(opAbs);
    PaddingAttr padding = getPadding(operandPaddings[0]);
    if (!padding) return nullptr;

    ArrayRef<int64_t> inLogical = padding.logical();
    ArrayRef<int64_t> outPadded = op.getResultType().getShape();

    SmallVector<int64_t> outLogical(outPadded.size(), 0);
    for (auto [inDim, group] : llvm::enumerate(op.getReassociationIndices())) {
      int64_t logicalProduct = inLogical[inDim];
      for (int64_t i = static_cast<int64_t>(group.size()) - 1; i >= 0; --i) {
        int64_t outDim = group[i];
        if (i == 0) {
          outLogical[outDim] = logicalProduct;
        } else {
          outLogical[outDim] = std::min(outPadded[outDim], logicalProduct);
          logicalProduct = (outLogical[outDim] > 0)
                               ? (logicalProduct / outLogical[outDim])
                               : 0;
        }
      }
    }
    return PaddingAttr::get(op.getContext(), outLogical, outPadded,
                            padding.isZeroPadded());
  }
};

struct CollapseShapePaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<
          CollapseShapePaddingImpl, tensor::CollapseShapeOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<tensor::CollapseShapeOp>(opAbs);
    PaddingAttr padding = getPadding(operandPaddings[0]);
    if (!padding) return nullptr;

    ArrayRef<int64_t> inLogical = padding.logical();
    ArrayRef<int64_t> inPadded = padding.padded();
    SmallVector<int64_t> outLogical, outPadded;
    for (ArrayRef<int64_t> group : op.getReassociationIndices()) {
      int64_t l = 1, p = 1;
      for (int64_t d : group) {
        l *= inLogical[d];
        p *= inPadded[d];
      }
      outLogical.push_back(l);
      outPadded.push_back(p);
    }
    return PaddingAttr::get(op.getContext(), outLogical, outPadded,
                            padding.isZeroPadded());
  }
};

struct GenericPaddingImpl
    : public PaddingSemanticsOpInterface::ExternalModel<GenericPaddingImpl,
                                                        linalg::GenericOp> {
  Attribute inferResultPadding(Operation* opAbs,
                               ArrayRef<Attribute> operandPaddings) const {
    auto op = cast<linalg::GenericOp>(opAbs);
    if (op.getNumParallelLoops() != op.getNumLoops() ||
        op.getNumDpsInits() != 1)
      return nullptr;
    auto resultType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (!resultType) return nullptr;

    ArrayRef<int64_t> outPadded = resultType.getShape();
    int64_t rank = outPadded.size();
    SmallVector<int64_t> outLogical(rank, -1);
    bool anyPadding = false;
    bool allZeroPadded = true;

    auto indexingMaps = op.getIndexingMapsArray();
    for (int i = 0; i < op.getNumDpsInputs(); ++i) {
      PaddingAttr padding = getPadding(operandPaddings[i]);
      if (!padding) continue;
      anyPadding = true;
      if (!padding.isZeroPadded()) allZeroPadded = false;

      AffineMap map = indexingMaps[i];
      for (unsigned pos = 0; pos < map.getNumResults(); ++pos) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(pos))) {
          unsigned outDim = dimExpr.getPosition();
          if (outDim < rank) {
            int64_t logVal = padding.logical()[pos];
            if (outLogical[outDim] == -1)
              outLogical[outDim] = logVal;
            else
              outLogical[outDim] = std::min(outLogical[outDim], logVal);
          }
        }
      }
    }
    if (!anyPadding) return nullptr;
    for (int d = 0; d < rank; ++d) {
      if (outLogical[d] == -1) outLogical[d] = outPadded[d];
    }
    return PaddingAttr::get(op.getContext(), outLogical, outPadded,
                            allZeroPadded);
  }
};

}  // namespace

void registerPaddingSemanticsInterfaces(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, tensor::TensorDialect*) {
    tensor::PadOp::attachInterface<PadOpPaddingImpl>(*ctx);
    tensor::ExpandShapeOp::attachInterface<ExpandShapePaddingImpl>(*ctx);
    tensor::CollapseShapeOp::attachInterface<CollapseShapePaddingImpl>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect*) {
    linalg::MatmulOp::attachInterface<MatmulPaddingImpl<linalg::MatmulOp>>(
        *ctx);
    linalg::BatchMatmulOp::attachInterface<
        MatmulPaddingImpl<linalg::BatchMatmulOp>>(*ctx);
    linalg::TransposeOp::attachInterface<TransposePaddingImpl>(*ctx);
    linalg::BroadcastOp::attachInterface<BroadcastPaddingImpl>(*ctx);
    linalg::ReduceOp::attachInterface<ReducePaddingImpl>(*ctx);
    linalg::GenericOp::attachInterface<GenericPaddingImpl>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect*) {
    arith::AddFOp::attachInterface<
        BinaryElementwisePaddingImpl<arith::AddFOp, false>>(*ctx);
    arith::SubFOp::attachInterface<
        BinaryElementwisePaddingImpl<arith::SubFOp, false>>(*ctx);
    arith::MulFOp::attachInterface<
        BinaryElementwisePaddingImpl<arith::MulFOp, true>>(*ctx);
    arith::NegFOp::attachInterface<
        UnaryElementwisePaddingImpl<arith::NegFOp, true>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, math::MathDialect*) {
    math::SqrtOp::attachInterface<
        UnaryElementwisePaddingImpl<math::SqrtOp, true>>(*ctx);
    math::ExpOp::attachInterface<
        UnaryElementwisePaddingImpl<math::ExpOp, false>>(*ctx);
    math::ErfOp::attachInterface<
        UnaryElementwisePaddingImpl<math::ErfOp, false>>(*ctx);
    math::RsqrtOp::attachInterface<
        UnaryElementwisePaddingImpl<math::RsqrtOp, false>>(*ctx);
    math::TanhOp::attachInterface<
        UnaryElementwisePaddingImpl<math::TanhOp, false>>(*ctx);
  });
}

// Builds a 0/1 dense constant over `type`: 1.0 at indices inside the
// trailing-pad logical region, `outside` selects whether the 1s mark the
// inside (mask) or the outside (pin) of the region.
DenseElementsAttr buildRegionIndicator(RankedTensorType type,
                                       ArrayRef<int64_t> logical,
                                       bool onesOutside, double oneValue) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t total = type.getNumElements();
  SmallVector<int64_t> idx(shape.size(), 0);
  SmallVector<APFloat> values;
  auto& semantics = cast<FloatType>(type.getElementType()).getFloatSemantics();
  values.reserve(total);
  for (int64_t linear = 0; linear < total; ++linear) {
    bool inPad = false;
    for (size_t d = 0; d < shape.size(); ++d) {
      if (idx[d] >= logical[d]) {
        inPad = true;
        break;
      }
    }
    bool one = onesOutside ? inPad : !inPad;
    if (one) {
      APFloat v(oneValue);
      bool losesInfo = false;
      v.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
      values.push_back(v);
    } else {
      values.push_back(APFloat::getZero(semantics));
    }
    for (int64_t d = shape.size() - 1; d >= 0; --d) {
      if (++idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
  return DenseElementsAttr::get(type, values);
}

tensor_ext::PaddingAttr getPaddingInfo(Value value) {
  Operation* def = value.getDefiningOp();
  if (!def) return nullptr;
  return def->getAttrOfType<tensor_ext::PaddingAttr>(kPaddingAttrName);
}

namespace {

struct PropagatePaddingPass
    : public impl::PropagatePaddingBase<PropagatePaddingPass> {
  void runOnOperation() override {
    DenseMap<Value, PaddingAttr> state;
    for (FunctionOpInterface func :
         getOperation()->getRegion(0).getOps<FunctionOpInterface>()) {
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (auto padding =
                func.getArgAttrOfType<PaddingAttr>(i, kPaddingAttrName)) {
          state[func.getArgument(i)] = padding;
          func.removeArgAttr(i, kPaddingAttrName);
        }
      }
    }
    WalkResult result =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
          // Transfer state into secret.generic bodies.
          if (auto generic = dyn_cast<secret::GenericOp>(op)) {
            for (auto [operand, arg] : llvm::zip(
                     generic.getInputs(), generic.getBody()->getArguments())) {
              if (PaddingAttr padding = state.lookup(operand))
                state[arg] = padding;
            }
            return WalkResult::advance();
          }

          Attribute inferred = nullptr;
          if (auto iface = dyn_cast<PaddingSemanticsOpInterface>(op)) {
            SmallVector<Attribute> operandPaddings;
            for (Value operand : op->getOperands())
              operandPaddings.push_back(state.lookup(operand));
            inferred = iface.inferResultPadding(operandPaddings);
          }
          if (!inferred)
            inferred = op->getAttrOfType<PaddingAttr>(kPaddingAttrName);
          if (!inferred) return WalkResult::advance();
          auto padding = cast<PaddingAttr>(inferred);

          // Invariant checks: the padded shape must be the actual type
          // shape, and the logical region must fit inside it.
          auto resultType =
              dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (!resultType || padding.padded() != resultType.getShape() ||
              padding.logical().size() != padding.padded().size()) {
            op->emitOpError()
                << "inferred padding " << padding
                << " does not match result type " << op->getResult(0).getType();
            return WalkResult::interrupt();
          }
          for (auto [l, p] : llvm::zip(padding.logical(), padding.padded())) {
            if (l > p || l < 0) {
              op->emitOpError() << "inferred padding " << padding
                                << " has logical shape exceeding padded shape";
              return WalkResult::interrupt();
            }
          }

          state[op->getResult(0)] = padding;
          if (!padding.isTrivial()) op->setAttr(kPaddingAttrName, padding);
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
