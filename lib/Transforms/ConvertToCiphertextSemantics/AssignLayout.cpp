#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/DynamicAPInt.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/FlatLinearValueConstraints.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using ::mlir::presburger::IntegerRelation;
using tensor_ext::LayoutAttr;

static std::string printRelation(const IntegerRelation& rel) {
  std::string str;
  llvm::raw_string_ostream os(str);
  rel.print(os);
  return str;
}

static FailureOr<Value> implementUnpackOpStep(
    Value input, LayoutAttr layout, Type targetType,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  IntegerRelation rel = layout.getIntegerRelation();
  LLVM_DEBUG(llvm::dbgs() << "implementUnpackOpStep with layout: " << layout
                          << " targetType: " << targetType << "\n");

  RankedTensorType unpackedTensorType = dyn_cast<RankedTensorType>(targetType);

  if (!unpackedTensorType) {
    // it's a scalar, so we can extract from any slot in the mapping
    std::vector<int64_t> point = anyRangePoint(rel);
    if (point.empty()) {
      return builder.emitError()
             << "Failed to find any point in the range of relation "
             << printRelation(rel);
    }

    SmallVector<Value> indices;
    for (int64_t idx : point) {
      indices.push_back(arith::ConstantIndexOp::create(builder, idx));
    }
    auto extractOp = tensor::ExtractOp::create(builder, input, indices);
    createdOpCallback(extractOp);
    return extractOp.getResult();
  }

  // Restrict the layout relation's domain bounds to the valid elements
  // [0, dimSize - 1] of targetType.
  for (unsigned i = 0; i < unpackedTensorType.getRank(); ++i) {
    if (unpackedTensorType.isDynamicDim(i)) continue;
    rel.addBound(presburger::BoundType::UB, i,
                 unpackedTensorType.getDimSize(i) - 1);
  }

  SmallVector<int> domainSchedule;
  for (unsigned i = 0; i < rel.getNumDomainVars(); ++i) {
    domainSchedule.push_back(i);
  }

  MLIRLoopNestGenerator generator(builder);
  auto loopNestCstr = generateLoopNestAsCStr(rel, domainSchedule);
  if (failed(loopNestCstr)) {
    return builder.emitError() << "Failed to generate loop nest for relation "
                               << printRelation(rel);
  }
  LLVM_DEBUG(llvm::dbgs() << "Generating loop nest assignment for relation "
                          << loopNestCstr.value() << "\n");

  TypedValue<RankedTensorType> ciphertextTensor =
      cast<TypedValue<RankedTensorType>>(input);
  RankedTensorType dataSemanticType = unpackedTensorType;

  auto dataTensor = cast<TypedValue<RankedTensorType>>(
      mlir::arith::ConstantOp::create(builder,
                                      builder.getZeroAttr(dataSemanticType))
          .getResult());
  auto loop = generator.generateForLoop(
      rel, {dataTensor},
      [&](mlir::OpBuilder& builder, Location loc, ValueRange exprs,
          ValueRange iterArgs) {
        SmallVector<Value> extractIndices;
        for (Value idx : exprs.drop_front(dataSemanticType.getRank())) {
          extractIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        // Extract from ciphertext and insert into dataTensor
        auto extracted = tensor::ExtractOp::create(
            builder, loc, ciphertextTensor, extractIndices);
        SmallVector<Value> insertIndices;
        for (Value idx :
             exprs.drop_back(ciphertextTensor.getType().getRank())) {
          insertIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        auto inserted = tensor::InsertOp::create(builder, loc, extracted,
                                                 iterArgs[0], insertIndices);
        return scf::ValueVector({inserted});
      },
      domainSchedule,
      /*reverse=*/true);
  if (failed(loop)) {
    return builder.emitError() << "Failed to generate loop nest for relation "
                               << printRelation(rel);
  }
  createdOpCallback(loop.value());
  return loop.value().getResults()[0];
}

static FailureOr<Value> implementAssignLayoutPermutation(
    Value input, DenseIntElementsAttr permutation, Type targetTypeTy,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  auto tensorType = dyn_cast<RankedTensorType>(input.getType());
  if (!tensorType)
    return builder.emitError() << "Permutation layout requires tensor input";

  if (tensorType.getRank() > 2)
    return builder.emitError() << "Arbitrary permutation layout only "
                                  "supports up to 2D tensors for now";

  RankedTensorType targetType = cast<RankedTensorType>(targetTypeTy);
  auto zeroCtxt = arith::ConstantOp::create(builder, targetType,
                                            builder.getZeroAttr(targetType));
  createdOpCallback(zeroCtxt);
  Value result = zeroCtxt.getResult();

  int64_t ctBound = (tensorType.getRank() == 2) ? tensorType.getDimSize(0) : 1;
  int64_t srcSlotBound = tensorType.getDimSize(tensorType.getRank() - 1);
  int64_t dstSlotBound = targetType.getDimSize(targetType.getRank() - 1);

  for (auto it = permutation.value_begin<APInt>();
       it != permutation.value_end<APInt>();) {
    int64_t srcCt = (*it++).getSExtValue();
    int64_t srcSlot = (*it++).getSExtValue();
    int64_t dstCt = (*it++).getSExtValue();
    int64_t dstSlot = (*it++).getSExtValue();

    if (srcCt >= ctBound || srcSlot >= srcSlotBound || dstCt >= ctBound ||
        dstSlot >= dstSlotBound) {
      return builder.emitError()
             << "Permutation index out of bounds: " << "src_ct=" << srcCt
             << ", src_slot=" << srcSlot << " (input bounds: ct < " << ctBound
             << ", slot < " << srcSlotBound << "); " << "dst_ct=" << dstCt
             << ", dst_slot=" << dstSlot << " (target bounds: ct < " << ctBound
             << ", slot < " << dstSlotBound << ")";
    }

    SmallVector<Value> extractIndices;

    if (tensorType.getRank() == 2) {
      auto srcCtIdxOp = arith::ConstantIndexOp::create(builder, srcCt);
      createdOpCallback(srcCtIdxOp);
      Value srcCtIdx = srcCtIdxOp.getResult();
      extractIndices.push_back(srcCtIdx);
    }
    auto srcSlotIdxOp = arith::ConstantIndexOp::create(builder, srcSlot);
    createdOpCallback(srcSlotIdxOp);
    extractIndices.push_back(srcSlotIdxOp.getResult());
    auto extracted = tensor::ExtractOp::create(builder, input, extractIndices);
    createdOpCallback(extracted);

    auto dstCtIdxOp = arith::ConstantIndexOp::create(builder, dstCt);
    createdOpCallback(dstCtIdxOp);
    auto dstSlotIdxOp = arith::ConstantIndexOp::create(builder, dstSlot);
    createdOpCallback(dstSlotIdxOp);
    auto insertOp = tensor::InsertOp::create(
        builder, extracted.getResult(), result,
        ValueRange{dstCtIdxOp.getResult(), dstSlotIdxOp.getResult()});
    createdOpCallback(insertOp);

    result = insertOp.getResult();
  }

  return result;
}

static FailureOr<Type> getIntermediateTargetType(
    Type elementType, LayoutAttr layout, ImplicitLocOpBuilder& builder) {
  presburger::IntegerRelation rel = layout.getIntegerRelation();
  unsigned numRangeVars = rel.getNumRangeVars();
  SmallVector<int64_t> targetShape;
  for (unsigned j = 0; j < numRangeVars; ++j) {
    auto ub = rel.getConstantBound64(
        presburger::BoundType::UB,
        rel.getVarKindOffset(presburger::VarKind::Range) + j);
    if (!ub.has_value()) {
      return builder.emitError() << "Unbounded range variable in relation";
    }
    targetShape.push_back(ub.value() + 1);
  }
  return RankedTensorType::get(targetShape, elementType);
}

static FailureOr<Value> implementCrtAssignLayoutStep(
    Value input, RankedTensorType inputType, RankedTensorType targetType,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  auto zeroOp = arith::ConstantOp::create(builder, targetType,
                                          builder.getZeroAttr(targetType));
  createdOpCallback(zeroOp);
  Value targetTensor = zeroOp.getResult();

  int64_t numSlots = targetType.getShape().back();
  int64_t numCts = (targetType.getRank() == 2) ? targetType.getDimSize(0) : 1;
  int64_t totalSlots = numCts * numSlots;

  Value c0 = arith::ConstantIndexOp::create(builder, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, 1);
  Value cTotalSlots = arith::ConstantIndexOp::create(builder, totalSlots);
  Value cNumSlots = arith::ConstantIndexOp::create(builder, numSlots);

  SmallVector<Value> dimSizes =
      llvm::map_to_vector(inputType.getShape(), [&](int64_t dim) -> Value {
        return arith::ConstantIndexOp::create(builder, dim);
      });

  auto loop = scf::ForOp::create(
      builder, c0, cTotalSlots, c1, ValueRange{targetTensor},
      [&](OpBuilder& b, Location loc, Value k, ValueRange iterArgs) {
        ImplicitLocOpBuilder ib(loc, b);

        SmallVector<Value> extractIndices =
            llvm::map_to_vector(dimSizes, [&](Value dimSize) -> Value {
              return arith::RemSIOp::create(ib, k, dimSize);
            });

        Value extracted = tensor::ExtractOp::create(ib, input, extractIndices);

        SmallVector<Value> insertIndices;
        if (targetType.getRank() == 2) {
          if (numCts == 1) {
            insertIndices.push_back(c0);
            insertIndices.push_back(k);
          } else {
            insertIndices.push_back(arith::DivSIOp::create(ib, k, cNumSlots));
            insertIndices.push_back(arith::RemSIOp::create(ib, k, cNumSlots));
          }
        } else {
          insertIndices.push_back(k);
        }

        Value inserted =
            tensor::InsertOp::create(ib, extracted, iterArgs[0], insertIndices);
        scf::YieldOp::create(ib, ValueRange{inserted});
      });
  createdOpCallback(loop);
  return loop.getResult(0);
}

static FailureOr<Value> implementAssignLayoutStep(
    Value input, LayoutAttr layout, Type targetTypeTy,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback, bool isLast,
    ArrayRef<int64_t> domainSchedule = {}) {
  presburger::IntegerRelation rel = layout.getIntegerRelation();
  RankedTensorType targetType = cast<RankedTensorType>(targetTypeTy);
  auto elementType = getElementTypeOrSelf(input.getType());
  unsigned numRangeVars = rel.getNumRangeVars();

  auto dataSemanticType = dyn_cast<RankedTensorType>(input.getType());
  if (dataSemanticType) {
    for (unsigned i = 0; i < dataSemanticType.getRank(); ++i) {
      rel.addBound(presburger::BoundType::LB,
                   rel.getVarKindOffset(presburger::VarKind::Domain) + i, 0);
      rel.addBound(presburger::BoundType::UB,
                   rel.getVarKindOffset(presburger::VarKind::Domain) + i,
                   dataSemanticType.getDimSize(i) - 1);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "implementAssignLayoutStep on input: "
                          << input.getType() << " with layout: " << layout
                          << " targetType: " << targetType << "\n");

  // If the input is a constant zero, then the output will be a constant zero
  // since we zero-fill slots.
  if (matchPattern(input, m_AnyZeroFloat()) || matchPattern(input, m_Zero())) {
    auto zeroOp = arith::ConstantOp::create(builder, targetType,
                                            builder.getZeroAttr(targetType));
    createdOpCallback(zeroOp);
    return zeroOp.getResult();
  }

  // The input came from an empty tensor, so we can just create an empty
  // ciphertext semantic tensor type.
  if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(input.getDefiningOp())) {
    auto emptyCiphertextOp = tensor::EmptyOp::create(
        builder, builder.getLoc(), targetType.getShape(),
        targetType.getElementType());
    createdOpCallback(emptyCiphertextOp);
    return emptyCiphertextOp.getResult();
  }

  // If the input has a bicyclic or tricyclic CRT layout, we directly compute
  // logical coordinates via modular remainder operations (arith.remsi) to
  // avoid ISL codegen overhead.
  int64_t numSlots = targetType.getShape().back();
  if (dataSemanticType &&
      (isRelationBicyclic(dataSemanticType, numSlots, rel) ||
       isRelationTricyclic(dataSemanticType, numSlots, rel))) {
    return implementCrtAssignLayoutStep(input, dataSemanticType, targetType,
                                        builder, createdOpCallback);
  }

  // The result can be simplified if the layout is dense in the ciphertext type,
  // and the input is a scalar or a constant splat.
  SplatElementsAttr splatAttr;
  bool inputIsScalar = !dataSemanticType;
  bool inputIsSplatConstant = matchPattern(input, m_Constant(&splatAttr));
  if ((inputIsScalar || inputIsSplatConstant) &&
      isDenseLayout(rel, targetType)) {
    // Regardless of being constant or not, a scalar can be splat into the
    // ciphertext tensor.
    if (inputIsScalar) {
      auto splatOp = tensor::SplatOp::create(builder, targetType, input);
      createdOpCallback(splatOp);
      return splatOp.getResult();
    }
    auto constantOp = arith::ConstantOp::create(
        builder, targetType,
        SplatElementsAttr::get(targetType,
                               splatAttr.getSplatValue<TypedAttr>()));
    createdOpCallback(constantOp);
    return constantOp.getResult();
  }

  // If the input is a dense/splat constant, evaluate the relation on its
  // elements at compile-time. This avoids generating a loop nest and evaluating
  // the packing at runtime. We gate this on isLast because we want to avoid
  // evaluating enumeratePoints on intermediate layouts.
  DenseElementsAttr constantAttr;
  if (matchPattern(input, m_Constant(&constantAttr)) && dataSemanticType &&
      isLast) {
    LLVM_DEBUG(llvm::dbgs() << "Detected constant input, evaluating layout\n");
    int64_t numTargetElements = targetType.getNumElements();
    Attribute zeroAttr = builder.getZeroAttr(elementType);
    SmallVector<Attribute> packedValues(numTargetElements, zeroAttr);

    PointPairCollector collector(dataSemanticType.getRank(),
                                 /*rangeDims=*/targetType.getRank());
    enumeratePoints(rel, collector);

    for (const auto& pointPair : collector.points) {
      const auto& domainPoint = pointPair.first;
      const auto& rangePoint = pointPair.second;

      Attribute valAttr;
      if (constantAttr.isSplat()) {
        valAttr = constantAttr.getSplatValue<Attribute>();
      } else {
        SmallVector<uint64_t> coord;
        coord.reserve(domainPoint.size());
        for (int64_t val : domainPoint) {
          coord.push_back(static_cast<uint64_t>(val));
        }
        valAttr =
            constantAttr
                .getValues<Attribute>()[mlir::ElementsAttr::getFlattenedIndex(
                    constantAttr.getType(), coord)];
      }
      auto flatIdx = getFlattenedIndex(targetType, llvm::to_vector(rangePoint));
      if (flatIdx >= 0 && flatIdx < numTargetElements) {
        packedValues[flatIdx] = valAttr;
      }
    }

    auto packedConstantAttr =
        DenseElementsAttr::get(targetType, ArrayRef<Attribute>(packedValues));
    auto constantOp = arith::ConstantOp::create(builder, builder.getLoc(),
                                                packedConstantAttr);
    createdOpCallback(constantOp);
    return constantOp.getResult();
  }

  auto zeroOp = arith::ConstantOp::create(builder, targetType,
                                          builder.getZeroAttr(targetType));
  createdOpCallback(zeroOp);
  Value targetTensor = zeroOp.getResult();

  MLIRLoopNestGenerator generator(builder, createdOpCallback);

  SmallVector<int> domainIndices = llvm::to_vector(llvm::map_range(
      domainSchedule, [](int64_t idx) { return static_cast<int>(idx); }));
  auto loopNestCstr = generateLoopNestAsCStr(rel, domainIndices);
  if (failed(loopNestCstr)) {
    return builder.emitError() << "Failed to generate loop nest for relation "
                               << printRelation(rel);
  }
  LLVM_DEBUG(llvm::dbgs() << "Generating loop nest assignment for relation "
                          << loopNestCstr.value() << "\n");

  auto loop = generator.generateForLoop(
      rel, {targetTensor},
      [&](OpBuilder& builder, Location loc, ValueRange exprs,
          ValueRange iterArgs) {
        Value extracted = input;
        auto dataSemanticType = dyn_cast<RankedTensorType>(input.getType());
        if (dataSemanticType) {
          SmallVector<Value> extractIndices;
          for (Value idx : exprs.take_front(dataSemanticType.getRank())) {
            extractIndices.push_back(arith::IndexCastOp::create(
                builder, loc, builder.getIndexType(), idx));
          }
          extracted =
              tensor::ExtractOp::create(builder, loc, input, extractIndices);
        }

        SmallVector<Value> insertIndices;
        unsigned relationDomainSize = rel.getNumDomainVars();
        for (Value idx : exprs.slice(relationDomainSize, numRangeVars)) {
          insertIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        auto inserted = tensor::InsertOp::create(builder, loc, extracted,
                                                 iterArgs[0], insertIndices);
        return scf::ValueVector({inserted});
      },
      domainIndices);

  if (failed(loop)) {
    return builder.emitError() << "Failed to generate loop nest";
  }

  createdOpCallback(loop.value());
  return loop.value().getResults()[0];
}

FailureOr<Value> implementAssignLayout(
    Value input, Attribute layout, int64_t ciphertextSize,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback,
    ArrayRef<int64_t> domainSchedule) {
  OpBuilder::InsertionGuard guard(builder);

  if (auto arrayAttr = dyn_cast<ArrayAttr>(layout)) {
    Value currentInput = input;
    for (size_t i = 0; i < arrayAttr.size(); ++i) {
      auto layoutAttr = dyn_cast<LayoutAttr>(arrayAttr[i]);
      if (!layoutAttr) {
        return builder.emitError()
               << "All layouts in the array must be LayoutAttr";
      }
      bool isLast = (i == arrayAttr.size() - 1);
      Type targetType;
      auto elementType = getElementTypeOrSelf(currentInput.getType());
      if (isLast) {
        targetType = materializeLayout(elementType, layoutAttr, ciphertextSize);
      } else {
        auto intermediateType =
            getIntermediateTargetType(elementType, layoutAttr, builder);
        if (failed(intermediateType)) return failure();
        targetType = intermediateType.value();
      }
      auto result =
          implementAssignLayoutStep(currentInput, layoutAttr, targetType,
                                    builder, createdOpCallback, isLast);
      if (failed(result)) {
        return failure();
      }
      currentInput = result.value();
    }
    return currentInput;
  }

  if (LayoutAttr layoutAttr = dyn_cast<LayoutAttr>(layout)) {
    auto elementType = getElementTypeOrSelf(input.getType());
    Type targetType =
        materializeLayout(elementType, layoutAttr, ciphertextSize);
    return implementAssignLayoutStep(input, layoutAttr, targetType, builder,
                                     createdOpCallback, /*isLast=*/true,
                                     domainSchedule);
  } else if (DenseIntElementsAttr elementAttr =
                 dyn_cast<DenseIntElementsAttr>(layout)) {
    Type targetType = materializePermutationLayout(input.getType(), elementAttr,
                                                   ciphertextSize);
    return implementAssignLayoutPermutation(input, elementAttr, targetType,
                                            builder, createdOpCallback);
  }
  return builder.emitError() << "Unsupported layout attribute type: " << layout;
}

FailureOr<Value> implementUnpackOp(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  OpBuilder::InsertionGuard guard(builder);
  if (auto layoutAttr = dyn_cast<LayoutAttr>(op.getLayout())) {
    return implementUnpackOpStep(op.getValue(), layoutAttr,
                                 op.getResult().getType(), builder,
                                 createdOpCallback);
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(op.getLayout())) {
    Value currentInput = op.getValue();
    Type finalTargetType = op.getResult().getType();

    for (int i = arrayAttr.size() - 1; i >= 0; --i) {
      auto layoutAttr = cast<LayoutAttr>(arrayAttr[i]);
      Type stepTargetType;
      if (i == 0) {
        stepTargetType = finalTargetType;
      } else {
        presburger::IntegerRelation rel = layoutAttr.getIntegerRelation();
        unsigned numDomainVars = rel.getNumDomainVars();
        SmallVector<int64_t> domainShape;
        for (unsigned d = 0; d < numDomainVars; ++d) {
          auto ub = rel.getConstantBound64(
              presburger::BoundType::UB,
              rel.getVarKindOffset(presburger::VarKind::Domain) + d);
          if (!ub.has_value()) {
            return builder.emitError()
                   << "Unbounded domain variable in relation";
          }
          domainShape.push_back(ub.value() + 1);
        }
        Type elementType = getElementTypeOrSelf(finalTargetType);
        stepTargetType = RankedTensorType::get(domainShape, elementType);
      }

      auto res = implementUnpackOpStep(currentInput, layoutAttr, stepTargetType,
                                       builder, createdOpCallback);
      if (failed(res)) {
        return failure();
      }
      currentInput = res.value();
    }
    return currentInput;
  }

  return builder.emitError()
         << "Unsupported layout attribute type: " << op.getLayout();
}

}  // namespace heir
}  // namespace mlir
