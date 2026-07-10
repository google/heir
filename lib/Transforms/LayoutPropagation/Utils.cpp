#include "lib/Transforms/LayoutPropagation/Utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
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
  std::unique_ptr<presburger::IntegerRelation> clonedRelation =
      inputLayout.getIntegerRelation().clone();

  auto offset = clonedRelation->getVarKindOffset(presburger::VarKind::Domain);
  for (int dim : llvm::reverse(dimsToReduce)) {
    // Set the dim to reduce equal to 0.
    auto dimIndex = offset + dim;
    assert(clonedRelation->getVarKindAt(dimIndex) ==
           presburger::VarKind::Domain);
    clonedRelation->setAndEliminate(dimIndex, 0);
  }

  MLIRContext* context = inputLayout.getContext();
  return LayoutAttr::getFromIntegerRelation(context,
                                            std::move(*clonedRelation));
}

}  // namespace heir
}  // namespace mlir
