#include "lib/Transforms/LinalgCanonicalizations/ConvPoolFusion.h"

#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/include/llvm/ADT/APFloat.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

std::optional<llvm::APFloat> extractFloatAttr(TypedAttr attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValue();
  }
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (denseAttr.isSplat() && isa<FloatType>(denseAttr.getElementType())) {
      return denseAttr.getSplatValue<llvm::APFloat>();
    }
  }
  return std::nullopt;
}

int64_t getEffectiveKernelHeight(const Conv2DSizeParams& conv,
                                 const Pool2DSizeParams& pool) {
  return (pool.pH - 1) * pool.dilationH * conv.strideH +
         (conv.kH - 1) * conv.dilationH + 1;
}

int64_t getEffectiveKernelWidth(const Conv2DSizeParams& conv,
                                const Pool2DSizeParams& pool) {
  return (pool.pW - 1) * pool.dilationW * conv.strideW +
         (conv.kW - 1) * conv.dilationW + 1;
}

namespace {

template <typename T>
llvm::SmallVector<T> fuseFiltersImpl(llvm::ArrayRef<T> convValues,
                                     const Conv2DSizeParams& conv,
                                     const Pool2DSizeParams& pool,
                                     int64_t kEffH, int64_t kEffW,
                                     std::function<T(const T&, const T&)> addFn,
                                     std::function<T(const T&)> scaleFn,
                                     const T& zeroVal) {
  int64_t F = conv.F;
  int64_t C = conv.C;
  int64_t kH = conv.kH;
  int64_t kW = conv.kW;
  int64_t pH = pool.pH;
  int64_t pW = pool.pW;

  int64_t strideConvH = conv.strideH;
  int64_t strideConvW = conv.strideW;
  int64_t dilationConvH = conv.dilationH;
  int64_t dilationConvW = conv.dilationW;

  int64_t dilationPoolH = pool.dilationH;
  int64_t dilationPoolW = pool.dilationW;

  int64_t numElements = F * C * kEffH * kEffW;
  llvm::SmallVector<T> fusedValues(numElements, zeroVal);

  for (int64_t f = 0; f < F; ++f) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t ph = 0; ph < pH; ++ph) {
        for (int64_t pw = 0; pw < pW; ++pw) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t u = ph * dilationPoolH * strideConvH + kh * dilationConvH;
              int64_t v = pw * dilationPoolW * strideConvW + kw * dilationConvW;

              int64_t srcIdx = ((f * C + c) * kH + kh) * kW + kw;
              int64_t dstIdx = ((f * C + c) * kEffH + u) * kEffW + v;

              T weight = convValues[srcIdx];
              T scaledWeight = scaleFn(weight);
              fusedValues[dstIdx] = addFn(fusedValues[dstIdx], scaledWeight);
            }
          }
        }
      }
    }
  }
  return fusedValues;
}

}  // namespace

llvm::SmallVector<llvm::APFloat> fuseFloatFilters(
    llvm::ArrayRef<llvm::APFloat> convValues, const Conv2DSizeParams& conv,
    const Pool2DSizeParams& pool, const llvm::fltSemantics& semantics,
    std::optional<llvm::APFloat> scaleVal) {
  int64_t kEffH = getEffectiveKernelHeight(conv, pool);
  int64_t kEffW = getEffectiveKernelWidth(conv, pool);

  llvm::APFloat scale = scaleVal.value_or(llvm::APFloat::getOne(semantics));
  auto addFn = [](const llvm::APFloat& a, const llvm::APFloat& b) {
    return a + b;
  };
  auto scaleFn = [&scale](const llvm::APFloat& w) { return w * scale; };

  return fuseFiltersImpl<llvm::APFloat>(convValues, conv, pool, kEffH, kEffW,
                                        addFn, scaleFn,
                                        llvm::APFloat::getZero(semantics));
}

llvm::SmallVector<llvm::APInt> fuseIntFilters(
    llvm::ArrayRef<llvm::APInt> convValues, const Conv2DSizeParams& conv,
    const Pool2DSizeParams& pool, unsigned width) {
  int64_t kEffH = getEffectiveKernelHeight(conv, pool);
  int64_t kEffW = getEffectiveKernelWidth(conv, pool);

  auto addFn = [](const llvm::APInt& a, const llvm::APInt& b) { return a + b; };
  auto scaleFn = [](const llvm::APInt& w) { return w; };

  return fuseFiltersImpl<llvm::APInt>(convValues, conv, pool, kEffH, kEffW,
                                      addFn, scaleFn, llvm::APInt(width, 0));
}

DenseElementsAttr fuseConv2DPoolingFilters(
    DenseElementsAttr convFilter, const Conv2DSizeParams& conv,
    const Pool2DSizeParams& pool, std::optional<llvm::APFloat> scaleVal) {
  auto filterType = cast<RankedTensorType>(convFilter.getType());
  Type elementType = filterType.getElementType();
  int64_t kEffH = getEffectiveKernelHeight(conv, pool);
  int64_t kEffW = getEffectiveKernelWidth(conv, pool);
  auto fusedFilterType =
      RankedTensorType::get({conv.F, conv.C, kEffH, kEffW}, elementType);

  return llvm::TypeSwitch<Type, DenseElementsAttr>(elementType)
      .Case<FloatType>([&](FloatType floatType) {
        auto convValues = convFilter.getValues<llvm::APFloat>();
        auto fusedValues =
            fuseFloatFilters(llvm::to_vector(convValues), conv, pool,
                             floatType.getFloatSemantics(), scaleVal);
        return DenseElementsAttr::get(fusedFilterType, fusedValues);
      })
      .Case<IntegerType>([&](IntegerType intType) {
        auto convValues = convFilter.getValues<llvm::APInt>();
        auto fusedValues = fuseIntFilters(llvm::to_vector(convValues), conv,
                                          pool, intType.getWidth());
        return DenseElementsAttr::get(fusedFilterType, fusedValues);
      })
      .Default([](Type) -> DenseElementsAttr {
        llvm_unreachable("unsupported element type for conv-pool fusion");
        return nullptr;
      });
}

FailureOr<ScalingFactorInfo> matchScalingFactor(
    Operation* user, Value poolResult, const llvm::fltSemantics& semantics) {
  auto getScale = [&](Value arg1,
                      Operation* op) -> std::optional<llvm::APFloat> {
    TypedAttr cstAttr;
    if (matchPattern(arg1, m_Constant(&cstAttr))) {
      if (auto valOpt = extractFloatAttr(cstAttr)) {
        llvm::APFloat val = *valOpt;
        if (isa<arith::DivFOp>(op)) {
          return llvm::APFloat::getOne(semantics) / val;
        } else {
          return val;
        }
      }
    }
    return std::nullopt;
  };

  // Case 1: arith.divf or arith.mulf on tensors
  if (isa<arith::DivFOp>(user) || isa<arith::MulFOp>(user)) {
    Value arg0 = user->getOperand(0);
    Value arg1 = user->getOperand(1);
    if (arg0 == poolResult) {
      if (auto scaleOpt = getScale(arg1, user)) {
        return ScalingFactorInfo{*scaleOpt, user};
      }
    }
  }

  // Case 2: linalg.generic
  if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
    if (genericOp.getNumDpsInputs() == 1 && genericOp.getNumDpsInits() == 1) {
      Block* body = genericOp.getBody();
      if (body->getOperations().size() == 2) {
        Operation& op = body->getOperations().front();
        if (isa<arith::DivFOp>(op) || isa<arith::MulFOp>(op)) {
          Value arg0 = op.getOperand(0);
          Value arg1 = op.getOperand(1);
          BlockArgument inputArg = body->getArgument(0);
          if (arg0 == inputArg) {
            if (auto scaleOpt = getScale(arg1, &op)) {
              return ScalingFactorInfo{*scaleOpt, genericOp};
            }
          }
        }
      }
    }
  }

  return failure();
}

}  // namespace heir
}  // namespace mlir
