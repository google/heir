#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>

#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/TransformUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

namespace {

bool hasBootstrapOp(func::FuncOp funcOp) {
  auto result = walkFuncAndCallees(funcOp, [&](Operation* op) {
    if (isa<BootOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

struct ConfigureCryptoContext
    : impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

  void runOnOperation() override {
    ModuleOp moduleOp = cast<ModuleOp>(getOperation());

    auto funcOp = detectEntryFunction(moduleOp, entryFunction);
    if (!funcOp) {
      moduleOp->emitError("Could not detect entry function");
      signalPassFailure();
      return;
    }

    auto paramSet = moduleOp->getAttrOfType<ParameterSetAttr>(
        cheddar::kParameterSetAttrName);
    auto schemeParam = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!paramSet && !schemeParam) {
      moduleOp->emitError(
          "Module missing either cheddar.parameter_set or ckks.schemeParam "
          "attribute");
      signalPassFailure();
      return;
    }

    int logN = paramSet ? paramSet.getLogN() : schemeParam.getLogN();
    int logScale =
        paramSet ? paramSet.getLogScale() : schemeParam.getLogDefaultScale();
    DenseI64ArrayAttr mainPrimes =
        paramSet ? paramSet.getMainPrimes() : schemeParam.getQ();
    DenseI64ArrayAttr auxPrimes =
        paramSet ? paramSet.getAuxPrimes() : schemeParam.getP();
    DenseI64ArrayAttr terminalPrimes =
        paramSet ? paramSet.getTerminalPrimes() : DenseI64ArrayAttr();
    DenseI32ArrayAttr levelConfig =
        paramSet ? paramSet.getLevelConfig() : DenseI32ArrayAttr();
    std::optional<int> wordBits =
        paramSet ? paramSet.getWordBits() : std::nullopt;
    DenseI32ArrayAttr additionalBase =
        paramSet ? paramSet.getAdditionalBase() : DenseI32ArrayAttr();

    int64_t maxChainLevel = 0;
    if (paramSet && paramSet.getLevelConfig()) {
      maxChainLevel = (paramSet.getLevelConfig().asArrayRef().size() / 2) - 1;
    } else {
      maxChainLevel = mainPrimes.asArrayRef().size() - 1;
    }

    bool hasBootstrap = hasBootstrapOp(funcOp);

    // 3-tier parameter resolution: default < cheddar.parameter_set < CLI
    // options
    std::optional<int> resolvedDefaultEncLevel = std::nullopt;
    if (paramSet && paramSet.getDefaultEncryptionLevel().has_value()) {
      resolvedDefaultEncLevel = paramSet.getDefaultEncryptionLevel();
    }
    if (defaultEncryptionLevel >= 0) {
      resolvedDefaultEncLevel = defaultEncryptionLevel.getValue();
    }

    std::optional<int> resolvedDenseWeight = std::nullopt;
    if (paramSet && paramSet.getDenseHammingWeight().has_value()) {
      resolvedDenseWeight = paramSet.getDenseHammingWeight();
    }
    if (denseHammingWeight > 0) {
      resolvedDenseWeight = denseHammingWeight.getValue();
    }

    std::optional<int> resolvedSparseWeight = std::nullopt;
    if (paramSet && paramSet.getSparseHammingWeight().has_value()) {
      resolvedSparseWeight = paramSet.getSparseHammingWeight();
    }
    if (sparseHammingWeight > 0) {
      resolvedSparseWeight = sparseHammingWeight.getValue();
    }

    std::optional<int> resolvedNumCts = std::nullopt;
    std::optional<int> resolvedNumStc = std::nullopt;
    if (hasBootstrap) {
      int cts = 3;
      if (paramSet && paramSet.getNumCtsLevels().has_value()) {
        cts = *paramSet.getNumCtsLevels();
      }
      if (numCtsLevels > 0) {
        cts = numCtsLevels.getValue();
      }
      resolvedNumCts = cts;

      int stc = 3;
      if (paramSet && paramSet.getNumStcLevels().has_value()) {
        stc = *paramSet.getNumStcLevels();
      }
      if (numStcLevels > 0) {
        stc = numStcLevels.getValue();
      }
      resolvedNumStc = stc;
    } else {
      if (paramSet && paramSet.getNumCtsLevels().has_value()) {
        resolvedNumCts = paramSet.getNumCtsLevels();
      }
      if (numCtsLevels > 0) {
        resolvedNumCts = numCtsLevels.getValue();
      }
      if (paramSet && paramSet.getNumStcLevels().has_value()) {
        resolvedNumStc = paramSet.getNumStcLevels();
      }
      if (numStcLevels > 0) {
        resolvedNumStc = numStcLevels.getValue();
      }
    }

    RotationAnalysis analysis;
    if (failed(analysis.run(moduleOp))) {
      moduleOp->emitError("Failed to run rotation analysis");
      signalPassFailure();
      return;
    }

    int64_t defaultLevel = resolvedDefaultEncLevel.has_value()
                               ? *resolvedDefaultEncLevel
                               : maxChainLevel;

    std::map<int64_t, int64_t> rotationMap;
    for (int64_t dist : analysis.getRotationIndices()) {
      rotationMap[dist] = prepareRotationKeysAtUseLevels ? -1 : defaultLevel;
    }
    if (prepareRotationKeysAtUseLevels) {
      moduleOp.walk([&](RotationOpInterface rotOp) {
        int64_t opLevel = defaultLevel;
        if (auto levelAttr = rotOp->getAttrOfType<IntegerAttr>("level")) {
          opLevel = levelAttr.getInt();
        }
        for (int64_t dist : analysis.getRotationIndices(rotOp)) {
          rotationMap[dist] = std::max(rotationMap[dist], opLevel);
        }
      });
      // Rotation indices discovered by RotationAnalysis via dynamic loop
      // simulation (or ops without an explicit level attribute) fall back to
      // defaultLevel.
      for (auto& [dist, lvl] : rotationMap) {
        if (lvl < 0) lvl = defaultLevel;
      }
    }

    MLIRContext* ctx = &getContext();
    Location loc = funcOp.getLoc();
    StringRef entryName = funcOp.getSymName();
    Type ctxType = hasBootstrap ? Type(BootContextType::get(ctx))
                                : Type(ContextType::get(ctx));
    auto ctxTensorType = RankedTensorType::get({}, ctxType);
    auto uiType = UserInterfaceType::get(ctx);
    auto uiTensorType = RankedTensorType::get({}, uiType);

    std::string setupName = (entryName + "__setup").str();
    std::string keygenName = (entryName + "__keygen").str();
    std::string configureName = (entryName + "__configure").str();

    auto effectiveParamSet = ParameterSetAttr::get(
        ctx, logN, logScale, mainPrimes, auxPrimes, terminalPrimes, levelConfig,
        wordBits, resolvedDefaultEncLevel, additionalBase, resolvedDenseWeight,
        resolvedSparseWeight, hasBootstrap, resolvedNumCts, resolvedNumStc);

    generateSetupFunc(moduleOp, loc, entryName, setupName, effectiveParamSet,
                      hasBootstrap, ctxType, ctxTensorType);
    generateKeygenFunc(moduleOp, loc, entryName, keygenName, logN, hasBootstrap,
                       rotationMap, ctxTensorType, uiType, uiTensorType);
    generateConfigureFunc(moduleOp, loc, setupName, keygenName, configureName,
                          ctxTensorType, uiTensorType);
  }

 private:
  void generateSetupFunc(ModuleOp moduleOp, Location loc, StringRef entryName,
                         StringRef setupName,
                         ParameterSetAttr effectiveParamSet, bool hasBootstrap,
                         Type ctxType, RankedTensorType ctxTensorType) {
    MLIRContext* ctx = moduleOp.getContext();
    OpBuilder builder = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto setupFuncType =
        FunctionType::get(ctx, /*inputs=*/{}, /*results=*/{ctxTensorType});
    auto setupFunc =
        builder.create<func::FuncOp>(loc, setupName, setupFuncType);
    setupFunc->setAttr("client.helper_func", builder.getUnitAttr());
    setupFunc->setAttr("client.interface_role",
                       builder.getStringAttr(kClientSetupRole));
    NamedAttrList heirInterfaceAttrs;
    heirInterfaceAttrs.set("func_name", builder.getStringAttr(entryName));
    heirInterfaceAttrs.set("roles",
                           builder.getStrArrayAttr({kClientSetupRole}));
    setupFunc->setAttr(kInterfaceAttrName,
                       heirInterfaceAttrs.getDictionary(ctx));

    Block* block = setupFunc.addEntryBlock();
    ImplicitLocOpBuilder b(loc, block, block->begin());

    auto paramType = ParameterType::get(ctx);
    auto params = MakeParameterOp::create(b, paramType, effectiveParamSet);

    auto emptyCtx =
        tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{}, ctxType);
    Value resultCtx;
    if (hasBootstrap) {
      auto numCtsAttr =
          b.getI64IntegerAttr(*effectiveParamSet.getNumCtsLevels());
      auto numStcAttr =
          b.getI64IntegerAttr(*effectiveParamSet.getNumStcLevels());
      resultCtx = CreateBootContextOp::create(
                      b, ctxTensorType, params, numCtsAttr, numStcAttr,
                      /*logMessageRatio=*/IntegerAttr(), emptyCtx)
                      .getResult()
                      .front();
    } else {
      resultCtx = CreateContextOp::create(b, ctxTensorType, params, emptyCtx)
                      .getResult()
                      .front();
    }
    func::ReturnOp::create(b, loc, resultCtx);
  }

  void generateKeygenFunc(ModuleOp moduleOp, Location loc, StringRef entryName,
                          StringRef keygenName, int logN, bool hasBootstrap,
                          const std::map<int64_t, int64_t>& rotationMap,
                          RankedTensorType ctxTensorType, Type uiType,
                          RankedTensorType uiTensorType) {
    MLIRContext* ctx = moduleOp.getContext();
    OpBuilder builder = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto keygenFuncType =
        FunctionType::get(ctx, /*inputs=*/{ctxTensorType},
                          /*results=*/{ctxTensorType, uiTensorType});
    auto keygenFunc =
        builder.create<func::FuncOp>(loc, keygenName, keygenFuncType);
    keygenFunc->setAttr("client.helper_func", builder.getUnitAttr());
    keygenFunc->setAttr("client.interface_role",
                        builder.getStringAttr(kClientKeygenRole));
    NamedAttrList heirInterfaceAttrs;
    heirInterfaceAttrs.set("func_name", builder.getStringAttr(entryName));
    heirInterfaceAttrs.set("roles",
                           builder.getStrArrayAttr({kClientKeygenRole}));
    keygenFunc->setAttr(kInterfaceAttrName,
                        heirInterfaceAttrs.getDictionary(ctx));

    Block* block = keygenFunc.addEntryBlock();
    ImplicitLocOpBuilder b(loc, block, block->begin());
    Value ctxArg = keygenFunc.getArgument(0);

    auto emptyUi = tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{}, uiType);
    Value uiVal =
        CreateUserInterfaceOp::create(b, uiTensorType, ctxArg, emptyUi)
            .getResult()
            .front();

    for (auto [dist, maxLevel] : rotationMap) {
      uiVal = PrepareRotKeyOp::create(b, uiTensorType, uiVal,
                                      b.getI64IntegerAttr(dist),
                                      b.getI64IntegerAttr(maxLevel))
                  .getResult()
                  .front();
    }

    if (hasBootstrap) {
      int64_t numSlots = 0;
      if (auto requestedSlotCountAttr = moduleOp->getAttrOfType<IntegerAttr>(
              kRequestedSlotCountAttrName)) {
        numSlots = requestedSlotCountAttr.getInt();
      } else if (auto actualSlotCountAttr =
                     moduleOp->getAttrOfType<IntegerAttr>(
                         kActualSlotCountAttrName)) {
        numSlots = actualSlotCountAttr.getInt();
      } else if (auto clientSlotCountAttr =
                     moduleOp->getAttrOfType<IntegerAttr>(
                         "client.requested_slot_count")) {
        numSlots = clientSlotCountAttr.getInt();
      } else {
        numSlots = 1LL << (logN - 1);
      }
      auto prepBoot = PrepareBootstrapOp::create(
          b, TypeRange{ctxTensorType, uiTensorType}, ctxArg, uiVal,
          b.getI64IntegerAttr(numSlots));
      func::ReturnOp::create(b, loc,
                             ValueRange{prepBoot.getCtxResult().front(),
                                        prepBoot.getUiResult().front()});
    } else {
      func::ReturnOp::create(b, loc, ValueRange{ctxArg, uiVal});
    }
  }

  void generateConfigureFunc(ModuleOp moduleOp, Location loc,
                             StringRef setupName, StringRef keygenName,
                             StringRef configureName,
                             RankedTensorType ctxTensorType,
                             RankedTensorType uiTensorType) {
    MLIRContext* ctx = moduleOp.getContext();
    OpBuilder builder = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto configureFuncType = FunctionType::get(
        ctx, /*inputs=*/{}, /*results=*/{ctxTensorType, uiTensorType});
    auto configureFunc =
        builder.create<func::FuncOp>(loc, configureName, configureFuncType);
    configureFunc->setAttr("client.helper_func", builder.getUnitAttr());

    Block* block = configureFunc.addEntryBlock();
    ImplicitLocOpBuilder b(loc, block, block->begin());

    auto setupCall =
        func::CallOp::create(b, loc, ctxTensorType,
                             SymbolRefAttr::get(ctx, setupName), ValueRange{});
    Value setupCtx = setupCall.getResult(0);

    auto keygenCall = func::CallOp::create(
        b, loc, TypeRange{ctxTensorType, uiTensorType},
        SymbolRefAttr::get(ctx, keygenName), ValueRange{setupCtx});
    Value retCtx = keygenCall.getResult(0);
    Value retUi = keygenCall.getResult(1);

    func::ReturnOp::create(b, loc, ValueRange{retCtx, retUi});
  }
};

}  // namespace
}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
