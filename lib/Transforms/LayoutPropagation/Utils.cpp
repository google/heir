#include "lib/Transforms/LayoutPropagation/Utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

Attribute makeKernelInfoAttr(MLIRContext* ctx, const KernelInfo& info) {
  SmallVector<NamedAttribute> attrs;
  attrs.reserve(3);
  attrs.push_back(
      NamedAttribute(StringAttr::get(ctx, kKernelShapeKey),
                     DenseI64ArrayAttr::get(ctx, info.resultShape)));
  if (!info.inputShape.empty()) {
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, kKernelInputShapeKey),
                       DenseI64ArrayAttr::get(ctx, info.inputShape)));
  }
  attrs.push_back(NamedAttribute(
      StringAttr::get(ctx, kGapFactorKey),
      IntegerAttr::get(IntegerType::get(ctx, 64), info.gapFactor)));
  return DictionaryAttr::get(ctx, attrs);
}

std::optional<KernelInfo> getKernelInfo(Attribute attr) {
  auto dictAttr = dyn_cast_or_null<DictionaryAttr>(attr);
  if (!dictAttr) return std::nullopt;
  KernelInfo info;
  if (auto inputShapeAttr =
          dictAttr.getAs<DenseI64ArrayAttr>(kKernelInputShapeKey)) {
    info.inputShape.assign(inputShapeAttr.asArrayRef().begin(),
                           inputShapeAttr.asArrayRef().end());
  }
  if (auto shapeAttr = dictAttr.getAs<DenseI64ArrayAttr>(kKernelShapeKey)) {
    info.resultShape.assign(shapeAttr.asArrayRef().begin(),
                            shapeAttr.asArrayRef().end());
  }
  if (auto gapFactorAttr = dictAttr.getAs<IntegerAttr>(kGapFactorKey)) {
    info.gapFactor = gapFactorAttr.getValue().getSExtValue();
  }
  return info;
}

std::optional<ConvMatrixOperand> foldConvSpatialPadding(
    RankedTensorType dataType, int64_t padding) {
  // Reject negative padding
  if (padding < 0) return std::nullopt;
  if (dataType.getRank() != 3 && dataType.getRank() != 4) return std::nullopt;
  SmallVector<int64_t> shape(dataType.getShape());
  // Dims 0 and 1 are (N, C); everything after them is spatial.
  for (int64_t dim = 2; dim < dataType.getRank(); ++dim) {
    shape[dim] -= 2 * padding;
    if (shape[dim] <= 0) return std::nullopt;
  }
  return ConvMatrixOperand{
      RankedTensorType::get(shape, dataType.getElementType()), padding};
}

int64_t getConvFoldedPadding(Operation* op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kConvFoldedPaddingAttrName)) {
    return attr.getInt();
  }
  return 0;
}

void setConvFoldedPadding(Operation* op, int64_t padding) {
  if (padding == 0) {
    // Drop an attribute an earlier run or an op clone left behind: a conv that
    // folded nothing must not be read as one that did.
    op->removeAttr(kConvFoldedPaddingAttrName);
    return;
  }
  op->setAttr(
      kConvFoldedPaddingAttrName,
      IntegerAttr::get(IntegerType::get(op->getContext(), 64), padding));
}

int64_t maxOfMaxes(ArrayRef<int64_t> d1, ArrayRef<int64_t> d2) {
  int64_t max = d1.front();
  for (int64_t di : d1) {
    max = std::max(max, di);
  }
  for (int64_t di : d2) {
    max = std::max(max, di);
  }
  return max;
}

SmallVector<int64_t> shiftByInserted(ArrayRef<int64_t> dims,
                                     ArrayRef<int64_t> inserts,
                                     bool increment) {
  SmallVector<int64_t> result;
  SmallVector<int64_t> sortedDims(dims);
  SmallVector<int64_t> sortedInserts(inserts);
  llvm::sort(sortedDims);
  llvm::sort(sortedInserts);

  int64_t shift = 0;
  auto dimIt = sortedDims.begin(), insertIt = sortedInserts.begin();
  while (dimIt != sortedDims.end()) {
    auto materializedDim = *dimIt + (increment ? shift : -shift);
    if (insertIt < sortedInserts.end() && *insertIt <= materializedDim) {
      ++insertIt;
      ++shift;
    } else {
      result.push_back(materializedDim);
      ++dimIt;
    }
  }

  return result;
}

SmallVector<int64_t> shiftByRemoved(ArrayRef<int64_t> dims,
                                    ArrayRef<int64_t> removed) {
  return shiftByInserted(dims, removed, false);
}

LayoutAttr convertLayoutForReduce(LayoutAttr inputLayout,
                                  ArrayRef<int64_t> dimsToReduce) {
  const presburger::IntegerRelation& rel = inputLayout.getIntegerRelation();
  unsigned domainOffset = rel.getVarKindOffset(presburger::VarKind::Domain);
  unsigned rangeOffset = rel.getVarKindOffset(presburger::VarKind::Range);
  unsigned numDomainVars = rel.getNumDomainVars();
  MLIRContext* context = inputLayout.getContext();

  // If the input layout has a tricyclic CRT layout (rank 3) and is reduced
  // along dimension 2:
  // Directly constructing the bicyclic relation for the remaining dimensions
  // avoids Fourier-Motzkin projection of the merged CRT modulo constraint,
  // which loses divisibility information and degenerates into an unconstrained
  // relation.
  auto slotUb =
      rel.getConstantBound64(presburger::BoundType::UB, rangeOffset + 1);
  if (slotUb.has_value() && numDomainVars == 3 &&
      dimsToReduce == ArrayRef<int64_t>{2}) {
    int64_t numSlots = slotUb.value() + 1;
    SmallVector<int64_t> shape;
    for (unsigned i = 0; i < 3; ++i) {
      auto ub =
          rel.getConstantBound64(presburger::BoundType::UB, domainOffset + i);
      if (ub.has_value()) {
        shape.push_back(ub.value() + 1);
      }
    }
    if (shape.size() == 3) {
      RankedTensorType tensorType =
          RankedTensorType::get(shape, Float32Type::get(context));
      if (isRelationTricyclic(tensorType, numSlots, rel)) {
        RankedTensorType reducedType = RankedTensorType::get(
            {tensorType.getDimSize(0), tensorType.getDimSize(1)},
            Float32Type::get(context));
        return LayoutAttr::getFromIntegerRelation(
            context, getBicyclicLayoutRelation(reducedType, numSlots));
      }
    }
  }

  // If the input layout has a bicyclic CRT layout (rank 2) and is reduced
  // along dimension 1:
  if (slotUb.has_value() && numDomainVars == 2 &&
      dimsToReduce == ArrayRef<int64_t>{1}) {
    int64_t numSlots = slotUb.value() + 1;
    SmallVector<int64_t> shape;
    for (unsigned i = 0; i < 2; ++i) {
      auto ub =
          rel.getConstantBound64(presburger::BoundType::UB, domainOffset + i);
      if (ub.has_value()) {
        shape.push_back(ub.value() + 1);
      }
    }
    if (shape.size() == 2) {
      RankedTensorType matrixType =
          RankedTensorType::get(shape, Float32Type::get(context));
      if (isRelationBicyclic(matrixType, numSlots, rel)) {
        presburger::IntegerRelation bicyclicRel =
            getBicyclicLayoutRelation(matrixType, numSlots);
        bicyclicRel.projectOut(1, 1);
        return LayoutAttr::getFromIntegerRelation(context,
                                                  std::move(bicyclicRel));
      }
    }
  }

  std::unique_ptr<presburger::IntegerRelation> clonedRelation =
      inputLayout.getIntegerRelation().clone();

  auto offset = clonedRelation->getVarKindOffset(presburger::VarKind::Domain);
  for (int dim : llvm::reverse(dimsToReduce)) {
    // Project out the reduced dimension.
    auto dimIndex = offset + dim;
    assert(clonedRelation->getVarKindAt(dimIndex) ==
           presburger::VarKind::Domain);
    clonedRelation->projectOut(dimIndex, 1);
  }

  return LayoutAttr::getFromIntegerRelation(context,
                                            std::move(*clonedRelation));
}

}  // namespace heir
}  // namespace mlir
