#include "lib/Dialect/LWE/Conversions/LWEToCheddarGPU/LWEToCheddarGPU.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOCHEDDARGPU
#include "lib/Dialect/LWE/Conversions/LWEToCheddarGPU/LWEToCheddarGPU.h.inc"

namespace {

struct CiphertextValue {
  int64_t level = 0;
  int64_t size = 2;
  int64_t degree = 1;
  std::string id;
  SmallVector<Value, 3> polys;
};

struct PlaintextValue {
  int64_t level = 0;
  int64_t degree = 1;
  bool isConst = false;
  double constVal = 0.0;
  std::string id;
  Value buffer;
};

bool isCiphertextType(Type type, lwe::LWECiphertextType& ctType) {
  if (auto direct = dyn_cast<lwe::LWECiphertextType>(type)) {
    ctType = direct;
    return true;
  }
  if (auto ranked = dyn_cast<RankedTensorType>(type)) {
    if (ranked.getRank() == 1 && ranked.getDimSize(0) == 1) {
      if (auto elem =
              dyn_cast<lwe::LWECiphertextType>(ranked.getElementType())) {
        ctType = elem;
        return true;
      }
    }
  }
  return false;
}

int64_t getCtLevel(lwe::LWECiphertextType ctType) {
  auto optLevel = lwe::getLevel(ctType);
  if (optLevel.has_value()) return optLevel.value();
  return 0;
}

int64_t getCtSize(lwe::LWECiphertextType ctType) {
  return ctType.getCiphertextSpace().getSize();
}

bool getSplatValue(Value val, double& splatVal) {
  Operation* defOp = val.getDefiningOp();
  if (!defOp) return false;
  if (auto cstOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(cstOp.getValue())) {
      if (denseAttr.isSplat()) {
        if (denseAttr.getElementType().isF32()) {
          splatVal = denseAttr.getSplatValue<FloatAttr>().getValueAsDouble();
          return true;
        }
        if (denseAttr.getElementType().isF64()) {
          splatVal = denseAttr.getSplatValue<FloatAttr>().getValueAsDouble();
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

struct LWEToCheddarGPUPass
    : public impl::LWEToCheddarGPUBase<LWEToCheddarGPUPass> {
  using impl::LWEToCheddarGPUBase<LWEToCheddarGPUPass>::LWEToCheddarGPUBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = &getContext();

    cheddar_gpu::ParamsAttr paramsAttr;
    if (failed(resolveParams(module, ctx, paramsAttr))) {
      signalPassFailure();
      return;
    }

    // Verify no unsupported operations exist.
    WalkResult rejectCheck = module.walk([&](Operation* op) {
      if (isa<ckks::BootstrapOp>(op)) {
        if (paramsAttr.getWord() != 64 || paramsAttr.getLogN() != 16 ||
            paramsAttr.getMain().size() != 26 || !paramsAttr.getTer().empty() ||
            paramsAttr.getAux().size() != 7 || paramsAttr.getAlpha() != 7) {
          op->emitOpError(
              "ckks.bootstrap is not supported without 64-bit logN=16 "
              "bootstrapping parameters (26 main, 0 ter, 7 aux, alpha=7)");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (rejectCheck.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Clean up client helpers and entry functions.
    SmallVector<func::FuncOp> toErase;
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (hasInterfaceRole(funcOp, kServerEvaluateRole)) {
        continue;
      }
      if (isClientHelper(funcOp) || hasInterfaceRole(funcOp, kEntryRole)) {
        toErase.push_back(funcOp);
      }
    }
    for (auto funcOp : toErase) {
      funcOp.erase();
    }

    // Lower remaining functions.
    for (auto funcOp :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (failed(lowerFunction(funcOp, paramsAttr))) {
        signalPassFailure();
        return;
      }
    }

    module->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
  }

 private:
  LogicalResult resolveParams(ModuleOp module, MLIRContext* ctx,
                              cheddar_gpu::ParamsAttr& resolvedParams) {
    auto existingSchemeParam = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);

    if (!paramsPath.empty()) {
      if (StringRef(paramsPath).ends_with(".yaml") ||
          StringRef(paramsPath).ends_with(".yml")) {
        auto loaded = ckks::loadSchemeParamAttrFromFile(
            ctx, paramsPath, [&]() { return module->emitError(); });
        if (failed(loaded)) return failure();
        resolvedParams = deriveParamsFromSchemeParam(ctx, *loaded);
      } else {
        auto p64 = cheddar_ref::Params<64>::LoadFromFile(paramsPath);
        resolvedParams = cheddar_gpu::ParamsAttr::fromParams<64>(ctx, p64);
      }

      if (existingSchemeParam) {
        auto existingQ = existingSchemeParam.getQ().asArrayRef();
        auto resolvedQ = resolvedParams.getMain().asArrayRef();
        auto existingP = existingSchemeParam.getP().asArrayRef();
        auto resolvedP = resolvedParams.getAux().asArrayRef();

        bool qMismatch = existingQ.size() > resolvedQ.size() ||
                         resolvedQ.take_front(existingQ.size()) != existingQ;
        bool pMismatch = existingP.size() > resolvedP.size() ||
                         resolvedP.take_front(existingP.size()) != existingP;

        if (static_cast<uint32_t>(existingSchemeParam.getLogN()) !=
                resolvedParams.getLogN() ||
            qMismatch || pMismatch) {
          module->emitOpError(
              "parameter mismatch between params and ckks.scheme_param");
          return failure();
        }
      }
      module->setAttr("cheddar_gpu.params", resolvedParams);
      return success();
    }

    if (auto existingParams = module->getAttrOfType<cheddar_gpu::ParamsAttr>(
            "cheddar_gpu.params")) {
      resolvedParams = existingParams;
      return success();
    }

    if (existingSchemeParam) {
      resolvedParams = deriveParamsFromSchemeParam(ctx, existingSchemeParam);
      module->setAttr("cheddar_gpu.params", resolvedParams);
      return success();
    }

    module->emitOpError(
        "module requires either cheddar_gpu.params, ckks.scheme_param, or "
        "params option");
    return failure();
  }

  cheddar_gpu::ParamsAttr deriveParamsFromSchemeParam(
      MLIRContext* ctx, ckks::SchemeParamAttr sp) {
    Builder builder(ctx);
    uint32_t word = 64;
    uint32_t logN = static_cast<uint32_t>(sp.getLogN());

    ArrayRef<int64_t> qRef = sp.getQ().asArrayRef();
    ArrayRef<int64_t> pRef = sp.getP().asArrayRef();
    SmallVector<int64_t> terVec;

    SmallVector<Attribute> levelConfig;
    for (size_t i = 0; i < qRef.size(); ++i) {
      levelConfig.push_back(builder.getArrayAttr({
          builder.getI32IntegerAttr(i + 1),
          builder.getI32IntegerAttr(0),
      }));
    }

    uint32_t alpha = pRef.size();
    uint32_t dnum = alpha > 0 ? (qRef.size() + alpha - 1) / alpha : 1;

    return cheddar_gpu::ParamsAttr::get(
        ctx, word, logN, DenseI64ArrayAttr::get(ctx, terVec),
        DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>(qRef)),
        DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>(pRef)),
        builder.getArrayAttr(levelConfig), alpha, dnum);
  }

  LogicalResult lowerFunction(func::FuncOp funcOp,
                              cheddar_gpu::ParamsAttr params) {
    MLIRContext* ctx = funcOp.getContext();
    OpBuilder builder(funcOp);
    int64_t N = 1LL << params.getLogN();
    Type wordType = builder.getI64Type();
    auto evalForm = cheddar_gpu::FormAttr::get(ctx, cheddar_gpu::Form::Eval);

    // Identify results
    FunctionType origType = funcOp.getFunctionType();
    SmallVector<std::pair<int64_t, int64_t>> resultSpecs;
    for (Type resType : origType.getResults()) {
      lwe::LWECiphertextType ctType;
      if (isCiphertextType(resType, ctType)) {
        resultSpecs.push_back({getCtLevel(ctType), getCtSize(ctType)});
      } else {
        funcOp->emitOpError("only ciphertext returns supported");
        return failure();
      }
    }

    // Identify input ciphertext arguments
    SmallVector<std::pair<int64_t, int64_t>> argSpecs;
    SmallVector<Value> origCtArgs;
    for (BlockArgument bArg : funcOp.getArguments()) {
      lwe::LWECiphertextType ctType;
      if (isCiphertextType(bArg.getType(), ctType)) {
        argSpecs.push_back({getCtLevel(ctType), getCtSize(ctType)});
        origCtArgs.push_back(bArg);
      }
    }

    // Collect rlwe_encode operations for hoisting in encode=args
    SmallVector<lwe::RLWEEncodeOp> encodeOps;
    funcOp.walk([&](lwe::RLWEEncodeOp encOp) { encodeOps.push_back(encOp); });

    // Also collect any plaintexts passed as function arguments
    SmallVector<BlockArgument> origPtArgs;
    for (BlockArgument bArg : funcOp.getArguments()) {
      if (isa<lwe::LWEPlaintextType>(bArg.getType())) {
        origPtArgs.push_back(bArg);
      }
    }

    // Determine degree of each encode / plaintext
    SmallVector<PlaintextValue> pts;
    int ptIdx = 0;
    for (lwe::RLWEEncodeOp encOp : encodeOps) {
      PlaintextValue ptVal;
      if (encOp.getLevel().has_value()) {
        ptVal.level = encOp.getLevel().value();
      } else {
        int64_t inferred = 0;
        for (Operation* user : encOp->getUsers()) {
          lwe::LWECiphertextType cty;
          Value ctVal;
          if (auto addP = dyn_cast<ckks::AddPlainOp>(user)) {
            ctVal = isCiphertextType(addP.getLhs().getType(), cty)
                        ? addP.getLhs()
                        : addP.getRhs();
          } else if (auto addP = dyn_cast<lwe::RAddPlainOp>(user)) {
            ctVal = isCiphertextType(addP.getLhs().getType(), cty)
                        ? addP.getLhs()
                        : addP.getRhs();
          } else if (auto subP = dyn_cast<ckks::SubPlainOp>(user)) {
            ctVal = isCiphertextType(subP.getLhs().getType(), cty)
                        ? subP.getLhs()
                        : subP.getRhs();
          } else if (auto subP = dyn_cast<lwe::RSubPlainOp>(user)) {
            ctVal = isCiphertextType(subP.getLhs().getType(), cty)
                        ? subP.getLhs()
                        : subP.getRhs();
          } else if (auto mulP = dyn_cast<ckks::MulPlainOp>(user)) {
            ctVal = isCiphertextType(mulP.getLhs().getType(), cty)
                        ? mulP.getLhs()
                        : mulP.getRhs();
          } else if (auto mulP = dyn_cast<lwe::RMulPlainOp>(user)) {
            ctVal = isCiphertextType(mulP.getLhs().getType(), cty)
                        ? mulP.getLhs()
                        : mulP.getRhs();
          }
          if (ctVal && isCiphertextType(ctVal.getType(), cty))
            inferred = getCtLevel(cty);
        }
        ptVal.level = inferred;
      }
      ptVal.degree = 1;
      double splatVal = 0.0;
      if (splatAsConst && getSplatValue(encOp.getInput(), splatVal)) {
        ptVal.isConst = true;
        ptVal.constVal = splatVal;
      }
      ptVal.id = "p" + std::to_string(ptIdx++);
      pts.push_back(ptVal);
    }

    for (BlockArgument bArg : origPtArgs) {
      PlaintextValue ptVal;
      // Infer level from user
      int64_t inferredLevel = 0;
      for (Operation* user : bArg.getUsers()) {
        lwe::LWECiphertextType cty;
        Value ctVal;
        if (auto addP = dyn_cast<ckks::AddPlainOp>(user)) {
          ctVal = isCiphertextType(addP.getLhs().getType(), cty)
                      ? addP.getLhs()
                      : addP.getRhs();
        } else if (auto addP = dyn_cast<lwe::RAddPlainOp>(user)) {
          ctVal = isCiphertextType(addP.getLhs().getType(), cty)
                      ? addP.getLhs()
                      : addP.getRhs();
        } else if (auto subP = dyn_cast<ckks::SubPlainOp>(user)) {
          ctVal = isCiphertextType(subP.getLhs().getType(), cty)
                      ? subP.getLhs()
                      : subP.getRhs();
        } else if (auto subP = dyn_cast<lwe::RSubPlainOp>(user)) {
          ctVal = isCiphertextType(subP.getLhs().getType(), cty)
                      ? subP.getLhs()
                      : subP.getRhs();
        } else if (auto mulP = dyn_cast<ckks::MulPlainOp>(user)) {
          ctVal = isCiphertextType(mulP.getLhs().getType(), cty)
                      ? mulP.getLhs()
                      : mulP.getRhs();
        } else if (auto mulP = dyn_cast<lwe::RMulPlainOp>(user)) {
          ctVal = isCiphertextType(mulP.getLhs().getType(), cty)
                      ? mulP.getLhs()
                      : mulP.getRhs();
        }
        if (ctVal && isCiphertextType(ctVal.getType(), cty)) {
          inferredLevel = getCtLevel(cty);
        }
      }
      ptVal.level = inferredLevel;
      ptVal.degree = 1;
      ptVal.id = "p" + std::to_string(ptIdx++);
      pts.push_back(ptVal);
    }

    // Pre-scan required keys (relinearization and rotations)
    bool needRelin = false;
    funcOp.walk([&](ckks::RelinearizeOp) { needRelin = true; });

    std::set<int64_t> rotDistSet;
    LogicalResult rotScanResult = success();
    funcOp.walk([&](ckks::RotateOp rotOp) {
      int64_t k = 0;
      if (rotOp.getStaticShift().has_value()) {
        k = rotOp.getStaticShift()->getInt();
      } else if (Value dynShift = rotOp.getDynamicShift()) {
        if (auto cstOp = dynShift.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue())) {
            k = intAttr.getInt();
          } else {
            rotOp.emitOpError("dynamic shift must be an integer constant");
            rotScanResult = failure();
            return WalkResult::interrupt();
          }
        } else {
          rotOp.emitOpError("dynamic shift must be a constant");
          rotScanResult = failure();
          return WalkResult::interrupt();
        }
      }
      int64_t slots = N / 2;
      int64_t r = ((k % slots) + slots) % slots;
      if (r > 0) rotDistSet.insert(r);
      return WalkResult::advance();
    });
    if (failed(rotScanResult)) return failure();
    SmallVector<int64_t> rotDists(rotDistSet.begin(), rotDistSet.end());

    // Pre-scan bootstrap ops
    bool hasBootstrap = false;
    int64_t bootNumSlots = -1;
    LogicalResult bootScanResult = success();
    funcOp.walk([&](ckks::BootstrapOp bootOp) {
      hasBootstrap = true;
      int64_t curSlots = N / 2;
      if (auto attr = bootOp->getAttrOfType<IntegerAttr>("cheddar.num_slots")) {
        curSlots = attr.getInt();
      } else if (auto attr =
                     funcOp->getAttrOfType<IntegerAttr>("cheddar.num_slots")) {
        curSlots = attr.getInt();
      }
      int64_t maxSlots = std::min<int64_t>(32768, N / 2);
      if (curSlots < 256 || curSlots > maxSlots ||
          (curSlots & (curSlots - 1)) != 0) {
        bootOp->emitOpError(
            "cheddar.num_slots must be a power of two in [256, min(32768, "
            "N/2)]");
        bootScanResult = failure();
        return WalkResult::interrupt();
      }
      if (bootNumSlots != -1 && bootNumSlots != curSlots) {
        bootOp->emitOpError(
            "multiple ckks.bootstrap ops with different num_slots");
        bootScanResult = failure();
        return WalkResult::interrupt();
      }
      bootNumSlots = curSlots;
      return WalkResult::advance();
    });
    if (failed(bootScanResult)) return failure();
    if (bootNumSlots == -1) bootNumSlots = N / 2;

    auto computeGaloisFactor = [&](int64_t r) -> int64_t {
      int64_t mod = 2 * N;
      int64_t base = 5 % mod;
      int64_t res = 1;
      int64_t exp = r;
      while (exp > 0) {
        if (exp & 1) res = (static_cast<__int128>(res) * base) % mod;
        base = (static_cast<__int128>(base) * base) % mod;
        exp >>= 1;
      }
      return res;
    };

    SmallVector<Attribute> keyAttrs;
    if (needRelin) {
      keyAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("type", builder.getStringAttr("relin")),
      }));
    }
    for (int64_t r : rotDists) {
      int64_t galois = computeGaloisFactor(r);
      keyAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("galois", builder.getI64IntegerAttr(galois)),
          builder.getNamedAttr("rot_dist", builder.getI64IntegerAttr(r)),
          builder.getNamedAttr("type", builder.getStringAttr("rot")),
      }));
    }

    uint32_t beta = params.getDnum();
    size_t numLimbs = params.getNumLimbs();
    auto evkMemRefType =
        MemRefType::get({static_cast<int64_t>(numLimbs), N}, wordType);

    // Build new function type: (outs..., args..., pts..., keys...) -> ()
    SmallVector<Type> newArgTypes;
    // Outs
    for (auto& [lvl, sz] : resultSpecs) {
      auto mType = MemRefType::get({lvl + 1, N}, wordType);
      for (int64_t k = 0; k < sz; ++k) {
        newArgTypes.push_back(mType);
      }
    }
    // Ins
    for (auto& [lvl, sz] : argSpecs) {
      auto mType = MemRefType::get({lvl + 1, N}, wordType);
      for (int64_t k = 0; k < sz; ++k) {
        newArgTypes.push_back(mType);
      }
    }
    // Pts
    for (auto& pt : pts) {
      if (pt.isConst) {
        newArgTypes.push_back(MemRefType::get({pt.level + 1}, wordType));
      } else {
        newArgTypes.push_back(MemRefType::get({pt.level + 1, N}, wordType));
      }
    }
    // Keys (for each key: beta bx digits, beta ax digits)
    for (size_t k = 0; k < keyAttrs.size(); ++k) {
      for (uint32_t d = 0; d < 2 * beta; ++d) {
        newArgTypes.push_back(evkMemRefType);
      }
    }
    // Boot slabs (consts_slab, evk_slab)
    if (hasBootstrap) {
      auto slabType =
          MemRefType::get({ShapedType::kDynamic}, builder.getI8Type());
      newArgTypes.push_back(slabType);
      newArgTypes.push_back(slabType);
    }

    FunctionType newFuncType = builder.getFunctionType(newArgTypes, {});
    funcOp.setType(newFuncType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());

    // Build abi attribute
    SmallVector<Attribute> resultAttrs;
    for (auto& [lvl, sz] : resultSpecs) {
      resultAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("level", builder.getI64IntegerAttr(lvl)),
          builder.getNamedAttr("size", builder.getI64IntegerAttr(sz)),
      }));
    }
    SmallVector<Attribute> argAttrs;
    for (auto& [lvl, sz] : argSpecs) {
      argAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("level", builder.getI64IntegerAttr(lvl)),
          builder.getNamedAttr("size", builder.getI64IntegerAttr(sz)),
      }));
    }
    SmallVector<Attribute> ptAttrs;
    for (auto& pt : pts) {
      ptAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("level", builder.getI64IntegerAttr(pt.level)),
          builder.getNamedAttr("degree", builder.getI64IntegerAttr(pt.degree)),
          builder.getNamedAttr("is_const", builder.getBoolAttr(pt.isConst)),
      }));
    }
    NamedAttrList initAbiAttrs;
    initAbiAttrs.append(
        builder.getNamedAttr("results", builder.getArrayAttr(resultAttrs)));
    initAbiAttrs.append(
        builder.getNamedAttr("args", builder.getArrayAttr(argAttrs)));
    initAbiAttrs.append(
        builder.getNamedAttr("pts", builder.getArrayAttr(ptAttrs)));
    initAbiAttrs.append(
        builder.getNamedAttr("keys", builder.getArrayAttr(keyAttrs)));
    if (hasBootstrap) {
      initAbiAttrs.append(builder.getNamedAttr(
          "boot", builder.getDictionaryAttr({builder.getNamedAttr(
                      "num_slots", builder.getI64IntegerAttr(bootNumSlots))})));
    }
    funcOp->setAttr("cheddar_gpu.abi",
                    initAbiAttrs.getDictionary(builder.getContext()));

    // Reconstruct block arguments
    Block& entryBlock = funcOp.getBody().front();
    // Old arguments
    SmallVector<Value> oldBlockArgs(entryBlock.getArguments().begin(),
                                    entryBlock.getArguments().end());

    // Add new arguments
    for (Type t : newArgTypes) {
      entryBlock.addArgument(t, funcOp.getLoc());
    }

    int newArgIdx = 0;
    // Map outs
    SmallVector<SmallVector<Value, 3>> outPolyBuffers;
    for (auto& [lvl, sz] : resultSpecs) {
      SmallVector<Value, 3> polys;
      for (int64_t k = 0; k < sz; ++k) {
        polys.push_back(
            entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++));
      }
      outPolyBuffers.push_back(polys);
    }

    // Map input cts
    DenseMap<Value, CiphertextValue> ctMap;
    std::string oracleTrace;
    llvm::raw_string_ostream traceOS(oracleTrace);

    int inArgIdx = 0;
    for (size_t i = 0; i < origCtArgs.size(); ++i) {
      Value oldArg = origCtArgs[i];
      auto& [lvl, sz] = argSpecs[i];
      CiphertextValue cv;
      cv.level = lvl;
      cv.size = sz;
      cv.degree = 1;
      cv.id = "arg" + std::to_string(inArgIdx++);
      for (int64_t k = 0; k < sz; ++k) {
        cv.polys.push_back(
            entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++));
      }
      ctMap[oldArg] = cv;
      traceOS << "{\"op\":\"arg\",\"id\":\"" << cv.id << "\",\"level\":" << lvl
              << ",\"size\":" << sz << "}\n";
    }

    // Map pts
    DenseMap<Operation*, Value> encMap;
    DenseMap<Value, std::string> ptIdMap;
    for (size_t i = 0; i < encodeOps.size(); ++i) {
      Value ptBuf = entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++);
      encMap[encodeOps[i].getOperation()] = ptBuf;
      pts[i].buffer = ptBuf;
      ptIdMap[ptBuf] = pts[i].id;
      traceOS << "{\"op\":\"encode\",\"id\":\"" << pts[i].id
              << "\",\"level\":" << pts[i].level
              << ",\"degree\":" << pts[i].degree
              << ",\"is_const\":" << (pts[i].isConst ? "true" : "false")
              << "}\n";
    }
    for (size_t i = 0; i < origPtArgs.size(); ++i) {
      Value ptBuf = entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++);
      pts[encodeOps.size() + i].buffer = ptBuf;
      ptIdMap[ptBuf] = pts[encodeOps.size() + i].id;
      // Map the old block argument
      ctMap[origPtArgs[i]] = CiphertextValue{pts[encodeOps.size() + i].level,
                                             1,
                                             1,
                                             pts[encodeOps.size() + i].id,
                                             {ptBuf}};
    }

    size_t ptInsertIdx = oldBlockArgs.size() + newArgIdx;

    // Map keys
    SmallVector<Value> relinEvkBx;
    SmallVector<Value> relinEvkAx;
    if (needRelin) {
      for (uint32_t d = 0; d < beta; ++d) {
        relinEvkBx.push_back(
            entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++));
      }
      for (uint32_t d = 0; d < beta; ++d) {
        relinEvkAx.push_back(
            entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++));
      }
    }

    DenseMap<int64_t, SmallVector<Value>> rotEvkBxMap;
    DenseMap<int64_t, SmallVector<Value>> rotEvkAxMap;
    for (int64_t r : rotDists) {
      SmallVector<Value> bxDigits;
      for (uint32_t d = 0; d < beta; ++d) {
        bxDigits.push_back(
            entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++));
      }
      rotEvkBxMap[r] = std::move(bxDigits);

      SmallVector<Value> axDigits;
      for (uint32_t d = 0; d < beta; ++d) {
        axDigits.push_back(
            entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++));
      }
      rotEvkAxMap[r] = std::move(axDigits);
    }

    Value bootConstsSlabArg;
    Value bootEvkSlabArg;
    if (hasBootstrap) {
      bootConstsSlabArg =
          entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++);
      bootEvkSlabArg =
          entryBlock.getArgument(oldBlockArgs.size() + newArgIdx++);
    }

    // Lower operations in entryBlock
    builder.setInsertionPointToStart(&entryBlock);
    int tempIdx = 0;

    auto makeTempId = [&]() { return "t" + std::to_string(tempIdx++); };

    SmallVector<Value> returnedValues;
    funcOp.walk([&](func::ReturnOp retOp) {
      for (Value operand : retOp.getOperands()) {
        returnedValues.push_back(operand);
      }
    });

    auto getDestPoly = [&](Value resVal, int64_t polyIdx, MemRefType mType,
                           Location loc) -> Value {
      Value curVal = resVal;
      while (true) {
        for (size_t r = 0; r < returnedValues.size(); ++r) {
          if (returnedValues[r] == curVal && r < outPolyBuffers.size() &&
              polyIdx < static_cast<int64_t>(outPolyBuffers[r].size())) {
            if (outPolyBuffers[r][polyIdx].getType() == mType) {
              return outPolyBuffers[r][polyIdx];
            }
          }
        }
        if (polyIdx >= 1 && curVal && curVal.hasOneUse()) {
          Operation* user = *curVal.user_begin();
          Value nextVal = nullptr;
          if (auto addPlain = dyn_cast<ckks::AddPlainOp>(user)) {
            nextVal = addPlain.getResult();
          } else if (auto raddPlain = dyn_cast<lwe::RAddPlainOp>(user)) {
            nextVal = raddPlain.getOutput();
          } else if (auto subPlain = dyn_cast<ckks::SubPlainOp>(user)) {
            if (subPlain.getLhs() == curVal) {
              nextVal = subPlain.getResult();
            }
          } else if (auto rsubPlain = dyn_cast<lwe::RSubPlainOp>(user)) {
            if (rsubPlain.getLhs() == curVal) {
              nextVal = rsubPlain.getOutput();
            }
          }
          if (nextVal) {
            curVal = nextVal;
            continue;
          }
        }
        break;
      }
      return memref::AllocOp::create(builder, loc, mType);
    };

    SmallVector<Operation*> toRemove;
    DenseSet<Operation*> fusedOps;

    for (Operation& op :
         llvm::make_early_inc_range(entryBlock.getOperations())) {
      builder.setInsertionPoint(&op);

      if (auto extractOp = dyn_cast<tensor::ExtractOp>(&op)) {
        if (ctMap.count(extractOp.getTensor())) {
          ctMap[extractOp.getResult()] = ctMap[extractOp.getTensor()];
          toRemove.push_back(&op);
          continue;
        }
      }

      if (auto encOp = dyn_cast<lwe::RLWEEncodeOp>(&op)) {
        toRemove.push_back(&op);
        continue;
      }

      Value addLhs, addRhs, addResult;
      bool isAdd = false;
      if (auto addOp = dyn_cast<ckks::AddOp>(&op)) {
        addLhs = addOp.getLhs();
        addRhs = addOp.getRhs();
        addResult = addOp.getResult();
        isAdd = true;
      } else if (auto raddOp = dyn_cast<lwe::RAddOp>(&op)) {
        addLhs = raddOp.getLhs();
        addRhs = raddOp.getRhs();
        addResult = raddOp.getOutput();
        isAdd = true;
      }
      if (isAdd) {
        CiphertextValue a = ctMap[addLhs];
        CiphertextValue b = ctMap[addRhs];
        int64_t lvl = a.level;
        int64_t sz = std::max(a.size, b.size);
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = sz;
        res.degree = a.degree;
        res.id = makeTempId();

        if (sz == 2 && a.size == 2 && b.size == 2) {
          Value outBx = getDestPoly(addResult, 0, mType, op.getLoc());
          Value outAx = getDestPoly(addResult, 1, mType, op.getLoc());
          res.polys.push_back(outBx);
          res.polys.push_back(outAx);
          cheddar_gpu::Add2Op::create(builder, op.getLoc(), outBx, outAx,
                                      a.polys[0], a.polys[1], b.polys[0],
                                      b.polys[1], evalForm, evalForm, evalForm,
                                      nullptr, nullptr, nullptr, nullptr);
        } else {
          for (int64_t k = 0; k < sz; ++k) {
            Value outBuf = getDestPoly(addResult, k, mType, op.getLoc());
            res.polys.push_back(outBuf);
            if (k < a.size && k < b.size) {
              cheddar_gpu::AddOp::create(
                  builder, op.getLoc(), outBuf, a.polys[k], b.polys[k],
                  evalForm, evalForm, evalForm, nullptr, nullptr, nullptr);
            } else if (k < a.size) {
              cheddar_gpu::CopyOp::create(builder, op.getLoc(), outBuf,
                                          a.polys[k], nullptr, nullptr);
            } else {
              cheddar_gpu::CopyOp::create(builder, op.getLoc(), outBuf,
                                          b.polys[k], nullptr, nullptr);
            }
          }
        }
        ctMap[addResult] = res;
        traceOS << "{\"op\":\"add\",\"out\":\"" << res.id << "\",\"lhs\":\""
                << a.id << "\",\"rhs\":\"" << b.id << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      Value subLhs, subRhs, subRes;
      bool isSub = false;
      if (auto subOp = dyn_cast<ckks::SubOp>(&op)) {
        subLhs = subOp.getLhs();
        subRhs = subOp.getRhs();
        subRes = subOp.getResult();
        isSub = true;
      } else if (auto rsubOp = dyn_cast<lwe::RSubOp>(&op)) {
        subLhs = rsubOp.getLhs();
        subRhs = rsubOp.getRhs();
        subRes = rsubOp.getOutput();
        isSub = true;
      }
      if (isSub) {
        CiphertextValue a = ctMap[subLhs];
        CiphertextValue b = ctMap[subRhs];
        int64_t lvl = a.level;
        int64_t sz = std::max(a.size, b.size);
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = sz;
        res.degree = a.degree;
        res.id = makeTempId();

        if (sz == 2 && a.size == 2 && b.size == 2) {
          Value outBx = getDestPoly(subRes, 0, mType, op.getLoc());
          Value outAx = getDestPoly(subRes, 1, mType, op.getLoc());
          res.polys.push_back(outBx);
          res.polys.push_back(outAx);
          cheddar_gpu::Sub2Op::create(builder, op.getLoc(), outBx, outAx,
                                      a.polys[0], a.polys[1], b.polys[0],
                                      b.polys[1], evalForm, evalForm, evalForm,
                                      nullptr, nullptr, nullptr, nullptr);
        } else {
          for (int64_t k = 0; k < sz; ++k) {
            Value outBuf = getDestPoly(subRes, k, mType, op.getLoc());
            res.polys.push_back(outBuf);
            if (k < a.size && k < b.size) {
              cheddar_gpu::SubOp::create(
                  builder, op.getLoc(), outBuf, a.polys[k], b.polys[k],
                  evalForm, evalForm, evalForm, nullptr, nullptr, nullptr);
            } else if (k < a.size) {
              cheddar_gpu::CopyOp::create(builder, op.getLoc(), outBuf,
                                          a.polys[k], nullptr, nullptr);
            } else {
              cheddar_gpu::NegOp::create(builder, op.getLoc(), outBuf,
                                         b.polys[k], evalForm, evalForm,
                                         nullptr, nullptr);
            }
          }
        }
        ctMap[subRes] = res;
        traceOS << "{\"op\":\"sub\",\"out\":\"" << res.id << "\",\"lhs\":\""
                << a.id << "\",\"rhs\":\"" << b.id << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      Value negIn, negRes;
      bool isNeg = false;
      if (auto negOp = dyn_cast<ckks::NegateOp>(&op)) {
        negIn = negOp.getOperand();
        negRes = negOp.getResult();
        isNeg = true;
      } else if (auto rnegOp = dyn_cast<lwe::RNegateOp>(&op)) {
        negIn = rnegOp.getInput();
        negRes = rnegOp.getOutput();
        isNeg = true;
      }
      if (isNeg) {
        CiphertextValue a = ctMap[negIn];
        int64_t lvl = a.level;
        int64_t sz = a.size;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = sz;
        res.degree = a.degree;
        res.id = makeTempId();

        if (sz == 2) {
          Value outBx = getDestPoly(negRes, 0, mType, op.getLoc());
          Value outAx = getDestPoly(negRes, 1, mType, op.getLoc());
          res.polys.push_back(outBx);
          res.polys.push_back(outAx);
          cheddar_gpu::Neg2Op::create(builder, op.getLoc(), outBx, outAx,
                                      a.polys[0], a.polys[1], evalForm,
                                      evalForm, nullptr, nullptr, nullptr);
        } else {
          for (int64_t k = 0; k < sz; ++k) {
            Value outBuf = getDestPoly(negRes, k, mType, op.getLoc());
            res.polys.push_back(outBuf);
            cheddar_gpu::NegOp::create(builder, op.getLoc(), outBuf, a.polys[k],
                                       evalForm, evalForm, nullptr, nullptr);
          }
        }
        ctMap[negRes] = res;
        traceOS << "{\"op\":\"neg\",\"out\":\"" << res.id << "\",\"in\":\""
                << a.id << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      Value addPLhs, addPRhs, addPRes;
      bool isAddPlain = false;
      if (auto addPlain = dyn_cast<ckks::AddPlainOp>(&op)) {
        addPLhs = addPlain.getLhs();
        addPRhs = addPlain.getRhs();
        addPRes = addPlain.getResult();
        isAddPlain = true;
      } else if (auto raddPlain = dyn_cast<lwe::RAddPlainOp>(&op)) {
        addPLhs = raddPlain.getLhs();
        addPRhs = raddPlain.getRhs();
        addPRes = raddPlain.getOutput();
        isAddPlain = true;
      }
      if (isAddPlain) {
        lwe::LWECiphertextType cty;
        bool ctIsLhs = isCiphertextType(addPLhs.getType(), cty);
        Value ctVal = ctIsLhs ? addPLhs : addPRhs;
        Value ptVal = ctIsLhs ? addPRhs : addPLhs;
        CiphertextValue a = ctMap[ctVal];
        Value ptBuf = ptVal;
        if (auto def = ptBuf.getDefiningOp<lwe::RLWEEncodeOp>()) {
          ptBuf = encMap[def.getOperation()];
        } else if (ctMap.count(ptBuf)) {
          ptBuf = ctMap[ptBuf].polys[0];
        }

        int64_t lvl = a.level;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = a.size;
        res.degree = a.degree;
        res.id = makeTempId();

        Value outBx = getDestPoly(addPRes, 0, mType, op.getLoc());
        res.polys.push_back(outBx);

        std::string ptId = ptIdMap.lookup(ptBuf);
        if (isa<MemRefType>(ptBuf.getType()) &&
            cast<MemRefType>(ptBuf.getType()).getRank() == 1) {
          // Splat constant
          cheddar_gpu::AddConstOp::create(builder, op.getLoc(), outBx,
                                          a.polys[0], ptBuf, evalForm, evalForm,
                                          nullptr, nullptr);
          traceOS << "{\"op\":\"add_const\",\"out\":\"" << res.id
                  << "\",\"in\":\"" << a.id << "\",\"level\":" << lvl
                  << ",\"pt\":\"" << ptId << "\"}\n";
        } else {
          cheddar_gpu::AddOp::create(builder, op.getLoc(), outBx, a.polys[0],
                                     ptBuf, evalForm, evalForm, evalForm,
                                     nullptr, nullptr, nullptr);
          traceOS << "{\"op\":\"add_plain\",\"out\":\"" << res.id
                  << "\",\"in\":\"" << a.id << "\",\"pt\":\"" << ptId
                  << "\"}\n";
        }

        // Copy remaining polys
        for (int64_t k = 1; k < a.size; ++k) {
          Value outPoly = getDestPoly(addPRes, k, mType, op.getLoc());
          res.polys.push_back(outPoly);
          if (a.polys[k] != outPoly) {
            cheddar_gpu::CopyOp::create(builder, op.getLoc(), outPoly,
                                        a.polys[k], nullptr, nullptr);
          }
        }

        ctMap[addPRes] = res;
        toRemove.push_back(&op);
        continue;
      }

      Value subPLhs, subPRhs, subPRes;
      bool isSubPlain = false;
      if (auto subPlain = dyn_cast<ckks::SubPlainOp>(&op)) {
        subPLhs = subPlain.getLhs();
        subPRhs = subPlain.getRhs();
        subPRes = subPlain.getResult();
        isSubPlain = true;
      } else if (auto rsubPlain = dyn_cast<lwe::RSubPlainOp>(&op)) {
        subPLhs = rsubPlain.getLhs();
        subPRhs = rsubPlain.getRhs();
        subPRes = rsubPlain.getOutput();
        isSubPlain = true;
      }
      if (isSubPlain) {
        lwe::LWECiphertextType cty;
        bool ctIsLhs = isCiphertextType(subPLhs.getType(), cty);
        Value ctVal = ctIsLhs ? subPLhs : subPRhs;
        Value ptVal = ctIsLhs ? subPRhs : subPLhs;
        CiphertextValue a = ctMap[ctVal];
        Value ptBuf = ptVal;
        if (auto def = ptBuf.getDefiningOp<lwe::RLWEEncodeOp>()) {
          ptBuf = encMap[def.getOperation()];
        } else if (ctMap.count(ptBuf)) {
          ptBuf = ctMap[ptBuf].polys[0];
        }

        int64_t lvl = a.level;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = a.size;
        res.degree = a.degree;
        res.id = makeTempId();

        Value outBx = getDestPoly(subPRes, 0, mType, op.getLoc());
        res.polys.push_back(outBx);

        bool isConst = (isa<MemRefType>(ptBuf.getType()) &&
                        cast<MemRefType>(ptBuf.getType()).getRank() == 1);

        if (ctIsLhs) {
          if (isConst) {
            cheddar_gpu::SubConstOp::create(builder, op.getLoc(), outBx,
                                            a.polys[0], ptBuf, evalForm,
                                            evalForm, nullptr, nullptr);
          } else {
            cheddar_gpu::SubOp::create(builder, op.getLoc(), outBx, a.polys[0],
                                       ptBuf, evalForm, evalForm, evalForm,
                                       nullptr, nullptr, nullptr);
          }
          for (int64_t k = 1; k < a.size; ++k) {
            Value outPoly = getDestPoly(subPRes, k, mType, op.getLoc());
            res.polys.push_back(outPoly);
            if (a.polys[k] != outPoly) {
              cheddar_gpu::CopyOp::create(builder, op.getLoc(), outPoly,
                                          a.polys[k], nullptr, nullptr);
            }
          }
        } else {
          // pt - ct
          if (isConst) {
            cheddar_gpu::SubOppositeConstOp::create(builder, op.getLoc(), outBx,
                                                    a.polys[0], ptBuf, evalForm,
                                                    evalForm, nullptr, nullptr);
          } else {
            cheddar_gpu::SubOp::create(builder, op.getLoc(), outBx, ptBuf,
                                       a.polys[0], evalForm, evalForm, evalForm,
                                       nullptr, nullptr, nullptr);
          }
          for (int64_t k = 1; k < a.size; ++k) {
            Value outPoly = getDestPoly(subPRes, k, mType, op.getLoc());
            res.polys.push_back(outPoly);
            cheddar_gpu::NegOp::create(builder, op.getLoc(), outPoly,
                                       a.polys[k], evalForm, evalForm, nullptr,
                                       nullptr);
          }
        }

        std::string ptId = ptIdMap.lookup(ptBuf);
        ctMap[subPRes] = res;
        traceOS << "{\"op\":\"sub_plain\",\"out\":\"" << res.id
                << "\",\"in\":\"" << a.id << "\",\"pt\":\"" << ptId << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      Value mulPLhs, mulPRhs, mulPRes;
      bool isMulPlain = false;
      if (auto mulPlain = dyn_cast<ckks::MulPlainOp>(&op)) {
        mulPLhs = mulPlain.getLhs();
        mulPRhs = mulPlain.getRhs();
        mulPRes = mulPlain.getResult();
        isMulPlain = true;
      } else if (auto rmulPlain = dyn_cast<lwe::RMulPlainOp>(&op)) {
        mulPLhs = rmulPlain.getLhs();
        mulPRhs = rmulPlain.getRhs();
        mulPRes = rmulPlain.getOutput();
        isMulPlain = true;
      }
      if (isMulPlain) {
        lwe::LWECiphertextType cty;
        bool ctIsLhs = isCiphertextType(mulPLhs.getType(), cty);
        Value ctVal = ctIsLhs ? mulPLhs : mulPRhs;
        Value ptVal = ctIsLhs ? mulPRhs : mulPLhs;
        CiphertextValue a = ctMap[ctVal];
        Value ptBuf = ptVal;
        if (auto def = ptBuf.getDefiningOp<lwe::RLWEEncodeOp>()) {
          ptBuf = encMap[def.getOperation()];
        } else if (ctMap.count(ptBuf)) {
          ptBuf = ctMap[ptBuf].polys[0];
        }

        int64_t lvl = a.level;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = a.size;
        res.degree = a.degree + 1;
        res.id = makeTempId();

        bool isConst = (isa<MemRefType>(ptBuf.getType()) &&
                        cast<MemRefType>(ptBuf.getType()).getRank() == 1);

        if (a.size == 2) {
          Value outBx = getDestPoly(mulPRes, 0, mType, op.getLoc());
          Value outAx = getDestPoly(mulPRes, 1, mType, op.getLoc());
          res.polys.push_back(outBx);
          res.polys.push_back(outAx);
          cheddar_gpu::CPAccum2Op::create(builder, op.getLoc(), outBx, outAx,
                                          /*ct_bx_srcs=*/ValueRange{a.polys[0]},
                                          /*ct_ax_srcs=*/ValueRange{a.polys[1]},
                                          /*common_srcs=*/ValueRange{ptBuf},
                                          evalForm,
                                          /*ct_forms=*/nullptr,
                                          /*common_forms=*/nullptr,
                                          /*out_limb_range=*/nullptr,
                                          /*prime_offset=*/nullptr,
                                          /*num_aux=*/nullptr);
        } else {
          for (int64_t k = 0; k < a.size; ++k) {
            Value outPoly = getDestPoly(mulPRes, k, mType, op.getLoc());
            res.polys.push_back(outPoly);
            if (isConst) {
              cheddar_gpu::MulConstOp::create(
                  builder, op.getLoc(), outPoly, a.polys[k], ptBuf, evalForm,
                  evalForm, nullptr, nullptr, nullptr);
            } else {
              cheddar_gpu::MulOp::create(builder, op.getLoc(), outPoly,
                                         a.polys[k], ptBuf, evalForm, evalForm,
                                         evalForm, nullptr, nullptr, nullptr);
            }
          }
        }

        std::string ptId = ptIdMap.lookup(ptBuf);
        ctMap[mulPRes] = res;
        traceOS << "{\"op\":\"mul_plain\",\"out\":\"" << res.id
                << "\",\"in\":\"" << a.id << "\",\"pt\":\"" << ptId << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      Value mulLhs, mulRhs, mulRes;
      bool isMul = false;
      if (auto mulOp = dyn_cast<ckks::MulOp>(&op)) {
        mulLhs = mulOp.getLhs();
        mulRhs = mulOp.getRhs();
        mulRes = mulOp.getResult();
        isMul = true;
      } else if (auto rmulOp = dyn_cast<lwe::RMulOp>(&op)) {
        mulLhs = rmulOp.getLhs();
        mulRhs = rmulOp.getRhs();
        mulRes = rmulOp.getOutput();
        isMul = true;
      }
      if (isMul) {
        CiphertextValue a = ctMap[mulLhs];
        CiphertextValue b = ctMap[mulRhs];
        int64_t lvl = a.level;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = 3;
        res.degree = a.degree + b.degree;
        res.id = makeTempId();

        Value outBx = getDestPoly(mulRes, 0, mType, op.getLoc());
        Value outAx = getDestPoly(mulRes, 1, mType, op.getLoc());
        Value outRx = getDestPoly(mulRes, 2, mType, op.getLoc());
        res.polys = {outBx, outAx, outRx};

        cheddar_gpu::TensorOp::create(
            builder, op.getLoc(), outBx, outAx, outRx, a.polys[0], a.polys[1],
            b.polys[0], b.polys[1], evalForm, evalForm, evalForm, nullptr);

        ctMap[mulRes] = res;
        traceOS << "{\"op\":\"mul\",\"out\":\"" << res.id << "\",\"lhs\":\""
                << a.id << "\",\"rhs\":\"" << b.id << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      if (auto relinOp = dyn_cast<ckks::RelinearizeOp>(&op)) {
        CiphertextValue a = ctMap[relinOp.getInput()];
        bool canFuse = false;
        ckks::RescaleOp targetRescaleOp;
        if (fuseRelinRescale && relinOp.getResult().hasOneUse()) {
          Operation* user = *relinOp.getResult().getUsers().begin();
          if (auto rescale = dyn_cast<ckks::RescaleOp>(user)) {
            canFuse = true;
            targetRescaleOp = rescale;
          }
        }

        if (canFuse) {
          int64_t lvl = a.level;
          int64_t targetLvl = lvl - 1;
          auto mType = MemRefType::get({targetLvl + 1, N}, wordType);

          CiphertextValue res;
          res.level = targetLvl;
          res.size = 2;
          res.degree = 1;
          res.id = makeTempId();

          Value outBx = getDestPoly(targetRescaleOp.getResult(), 0, mType,
                                    targetRescaleOp.getLoc());
          Value outAx = getDestPoly(targetRescaleOp.getResult(), 1, mType,
                                    targetRescaleOp.getLoc());
          res.polys = {outBx, outAx};

          cheddar_gpu::RelinearizeRescaleOp::create(
              builder, op.getLoc(), outBx, outAx, a.polys[0], a.polys[1],
              a.polys[2], relinEvkBx, relinEvkAx,
              builder.getI64IntegerAttr(lvl), evalForm, evalForm);

          ctMap[targetRescaleOp.getResult()] = res;
          traceOS << "{\"op\":\"relinearize_rescale\",\"out\":\"" << res.id
                  << "\",\"in\":\"" << a.id << "\"}\n";
          fusedOps.insert(targetRescaleOp.getOperation());
          toRemove.push_back(&op);
          continue;
        }

        int64_t lvl = a.level;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = lvl;
        res.size = 2;
        res.degree = 1;
        res.id = makeTempId();

        Value outBx = getDestPoly(relinOp.getResult(), 0, mType, op.getLoc());
        Value outAx = getDestPoly(relinOp.getResult(), 1, mType, op.getLoc());
        res.polys = {outBx, outAx};

        cheddar_gpu::RelinearizeOp::create(
            builder, op.getLoc(), outBx, outAx, a.polys[0], a.polys[1],
            a.polys[2], relinEvkBx, relinEvkAx, builder.getI64IntegerAttr(lvl),
            evalForm, evalForm);

        ctMap[relinOp.getResult()] = res;
        traceOS << "{\"op\":\"relinearize\",\"out\":\"" << res.id
                << "\",\"in\":\"" << a.id << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      if (auto rescaleOp = dyn_cast<ckks::RescaleOp>(&op)) {
        if (fusedOps.count(rescaleOp.getOperation())) {
          toRemove.push_back(&op);
          continue;
        }

        CiphertextValue a = ctMap[rescaleOp.getOperand()];
        int64_t lvl = a.level;
        int64_t targetLvl = lvl - 1;
        auto mType = MemRefType::get({targetLvl + 1, N}, wordType);

        CiphertextValue res;
        res.level = targetLvl;
        res.size = a.size;
        res.degree = a.degree - 1;
        res.id = makeTempId();

        for (int64_t k = 0; k < a.size; ++k) {
          Value outPoly =
              getDestPoly(rescaleOp.getResult(), k, mType, op.getLoc());
          res.polys.push_back(outPoly);
          cheddar_gpu::RescaleOp::create(
              builder, op.getLoc(), outPoly, a.polys[k],
              builder.getI64IntegerAttr(lvl), evalForm, evalForm);
        }

        ctMap[rescaleOp.getResult()] = res;
        traceOS << "{\"op\":\"rescale\",\"out\":\"" << res.id << "\",\"in\":\""
                << a.id << "\"}\n";
        toRemove.push_back(&op);
        continue;
      }

      if (auto rotOp = dyn_cast<ckks::RotateOp>(&op)) {
        CiphertextValue a = ctMap[rotOp.getInput()];
        int64_t lvl = a.level;
        auto mType = MemRefType::get({lvl + 1, N}, wordType);

        int64_t k = 0;
        if (rotOp.getStaticShift().has_value()) {
          k = rotOp.getStaticShift()->getInt();
        } else if (Value dynShift = rotOp.getDynamicShift()) {
          if (auto cstOp = dynShift.getDefiningOp<arith::ConstantOp>()) {
            k = cast<IntegerAttr>(cstOp.getValue()).getInt();
          }
        }

        int64_t slots = N / 2;
        int64_t r = ((k % slots) + slots) % slots;

        CiphertextValue res;
        res.level = lvl;
        res.size = a.size;
        res.degree = a.degree;
        res.id = makeTempId();

        Value outBx = getDestPoly(rotOp.getResult(), 0, mType, op.getLoc());
        Value outAx = getDestPoly(rotOp.getResult(), 1, mType, op.getLoc());
        res.polys = {outBx, outAx};

        if (r == 0) {
          if (a.polys[0] != outBx) {
            cheddar_gpu::CopyOp::create(builder, op.getLoc(), outBx, a.polys[0],
                                        nullptr, nullptr);
          }
          if (a.polys[1] != outAx) {
            cheddar_gpu::CopyOp::create(builder, op.getLoc(), outAx, a.polys[1],
                                        nullptr, nullptr);
          }
          traceOS << "{\"op\":\"copy\",\"out\":\"" << res.id << "\",\"in\":\""
                  << a.id << "\"}\n";
        } else {
          int64_t galoisFactor = computeGaloisFactor(r);
          cheddar_gpu::HRotOp::create(
              builder, op.getLoc(), outBx, outAx, a.polys[0], a.polys[1],
              rotEvkBxMap[r], rotEvkAxMap[r], builder.getI64IntegerAttr(lvl),
              builder.getI64IntegerAttr(r),
              builder.getI64IntegerAttr(galoisFactor), evalForm, evalForm);
          traceOS << "{\"op\":\"hrot\",\"out\":\"" << res.id << "\",\"in\":\""
                  << a.id << "\",\"rot_dist\":" << r << "}\n";
        }

        ctMap[rotOp.getResult()] = res;
        toRemove.push_back(&op);
        continue;
      }

      if (auto bootOp = dyn_cast<ckks::BootstrapOp>(&op)) {
        CiphertextValue a = ctMap[bootOp.getInput()];
        if (a.size != 2 || a.level < 0 || a.level > 13) {
          bootOp.emitOpError(
              "input ciphertext must have size 2 and level in [0, 13]");
          return failure();
        }

        int64_t outLvl = -1;
        if (bootOp.getTargetLevel().has_value()) {
          outLvl = bootOp.getTargetLevel().value();
        } else {
          lwe::LWECiphertextType cty;
          if (isCiphertextType(bootOp.getResult().getType(), cty)) {
            outLvl = getCtLevel(cty);
          }
        }
        if (outLvl < 0 || outLvl > 10) {
          bootOp.emitOpError("target level must be in [0, 10]");
          return failure();
        }

        auto outMType = MemRefType::get({outLvl + 1, N}, wordType);
        Value outBx = getDestPoly(bootOp.getResult(), 0, outMType, op.getLoc());
        Value outAx = getDestPoly(bootOp.getResult(), 1, outMType, op.getLoc());

        cheddar_gpu::BootOp::create(
            builder, op.getLoc(), outBx, outAx, a.polys[0], a.polys[1],
            bootConstsSlabArg, bootEvkSlabArg,
            builder.getI64IntegerAttr(bootNumSlots), evalForm, evalForm,
            /*params=*/cheddar_gpu::ParamsAttr{});

        CiphertextValue res{outLvl, 2, 1, makeTempId(), {outBx, outAx}};
        ctMap[bootOp.getResult()] = res;
        traceOS << "{\"op\":\"bootstrap\",\"out\":\"" << res.id
                << "\",\"in\":\"" << a.id << "\",\"num_slots\":" << bootNumSlots
                << ",\"target_level\":" << outLvl << "}\n";
        toRemove.push_back(&op);
        continue;
      }

      if (auto levelReduceOp = dyn_cast<ckks::LevelReduceOp>(&op)) {
        // Per Context.cpp LevelDown:
        // level_reduce ≡ Context::LevelDown = n *
        // [mul_const(EncodeConstant(1.0, L)); rescale]
        CiphertextValue a = ctMap[levelReduceOp.getOperand()];
        int64_t drop = levelReduceOp.getLevelToDrop();
        int64_t currentLvl = a.level;
        CiphertextValue curr = a;

        for (int64_t d = 0; d < drop; ++d) {
          int64_t l = currentLvl - d;
          int64_t nextL = l - 1;
          auto nextMType = MemRefType::get({nextL + 1, N}, wordType);
          auto currMType = MemRefType::get({l + 1, N}, wordType);

          // Hoist a "one" constant for this level
          PlaintextValue onePt;
          onePt.level = l;
          onePt.degree = 1;
          onePt.isConst = true;
          onePt.constVal = 1.0;
          onePt.id = "one_L" + std::to_string(l);

          // Add to function arguments
          auto constMemRefType = MemRefType::get({l + 1}, wordType);
          Value oneArg = entryBlock.insertArgument(
              ptInsertIdx++, constMemRefType, op.getLoc());
          onePt.buffer = oneArg;
          pts.push_back(onePt);

          // Multiply all polys by one
          SmallVector<Value, 3> mulPolys;
          if (curr.size == 2) {
            Value mulBx =
                memref::AllocOp::create(builder, op.getLoc(), currMType);
            Value mulAx =
                memref::AllocOp::create(builder, op.getLoc(), currMType);
            cheddar_gpu::CPAccum2Op::create(
                builder, op.getLoc(), mulBx, mulAx,
                /*ct_bx_srcs=*/ValueRange{curr.polys[0]},
                /*ct_ax_srcs=*/ValueRange{curr.polys[1]},
                /*common_srcs=*/ValueRange{oneArg}, evalForm,
                /*ct_forms=*/nullptr,
                /*common_forms=*/nullptr,
                /*out_limb_range=*/nullptr,
                /*prime_offset=*/nullptr,
                /*num_aux=*/nullptr);
            mulPolys.push_back(mulBx);
            mulPolys.push_back(mulAx);
          } else {
            for (int64_t k = 0; k < curr.size; ++k) {
              Value mulOut =
                  memref::AllocOp::create(builder, op.getLoc(), currMType);
              cheddar_gpu::MulConstOp::create(
                  builder, op.getLoc(), mulOut, curr.polys[k], oneArg, evalForm,
                  evalForm, nullptr, nullptr, nullptr);
              mulPolys.push_back(mulOut);
            }
          }

          // Rescale all polys to nextL
          CiphertextValue nextVal;
          nextVal.level = nextL;
          nextVal.size = curr.size;
          nextVal.degree = curr.degree;
          nextVal.id = makeTempId();

          for (int64_t k = 0; k < curr.size; ++k) {
            Value outPoly;
            if (d == drop - 1) {
              outPoly = getDestPoly(levelReduceOp.getResult(), k, nextMType,
                                    op.getLoc());
            } else {
              outPoly =
                  memref::AllocOp::create(builder, op.getLoc(), nextMType);
            }
            nextVal.polys.push_back(outPoly);
            cheddar_gpu::RescaleOp::create(
                builder, op.getLoc(), outPoly, mulPolys[k],
                builder.getI64IntegerAttr(l), evalForm, evalForm);
          }
          curr = nextVal;
        }

        ctMap[levelReduceOp.getResult()] = curr;
        traceOS << "{\"op\":\"level_down\",\"out\":\"" << curr.id
                << "\",\"in\":\"" << a.id
                << "\",\"target\":" << (a.level - drop) << "}\n";
        toRemove.push_back(&op);
        continue;
      }

      if (auto insertOp = dyn_cast<tensor::InsertOp>(&op)) {
        if (ctMap.count(insertOp.getScalar())) {
          ctMap[insertOp.getResult()] = ctMap[insertOp.getScalar()];
          toRemove.push_back(&op);
          continue;
        }
      }

      if (auto retOp = dyn_cast<func::ReturnOp>(&op)) {
        SmallVector<std::string> retIds;
        for (size_t r = 0; r < retOp.getNumOperands(); ++r) {
          Value retVal = retOp.getOperand(r);
          if (ctMap.count(retVal)) {
            CiphertextValue cv = ctMap[retVal];
            retIds.push_back(cv.id);
            for (int64_t k = 0; k < cv.size; ++k) {
              if (cv.polys[k] != outPolyBuffers[r][k]) {
                cheddar_gpu::CopyOp::create(builder, op.getLoc(),
                                            outPolyBuffers[r][k], cv.polys[k],
                                            nullptr, nullptr);
              }
            }
          }
        }
        traceOS << "{\"op\":\"ret\",\"ids\":[";
        for (size_t i = 0; i < retIds.size(); ++i) {
          if (i > 0) traceOS << ",";
          traceOS << "\"" << retIds[i] << "\"";
        }
        traceOS << "]}\n";
        traceOS << "{\"op\":\"keys\",\"relin\":"
                << (needRelin ? "true" : "false") << ",\"rot\":[";
        for (size_t i = 0; i < rotDists.size(); ++i) {
          if (i > 0) traceOS << ",";
          traceOS << rotDists[i];
        }
        traceOS << "]}\n";

        func::ReturnOp::create(builder, op.getLoc());
        toRemove.push_back(&op);
        continue;
      }
    }

    // Remove old ops in reverse order so users are destroyed before defs
    for (Operation* op : llvm::reverse(toRemove)) {
      op->erase();
    }

    // Erase old block arguments now that all their uses have been removed
    for (int i = static_cast<int>(oldBlockArgs.size()) - 1; i >= 0; --i) {
      entryBlock.eraseArgument(i);
    }

    // Update funcOp signature and ABI to match final entryBlock argument types
    funcOp.setType(builder.getFunctionType(entryBlock.getArgumentTypes(), {}));
    funcOp->removeAttr("arg_attrs");
    funcOp->removeAttr("res_attrs");
    funcOp->removeAttr("heir.interface");
    SmallVector<Attribute> finalPtAttrs;
    for (auto& pt : pts) {
      finalPtAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("level", builder.getI64IntegerAttr(pt.level)),
          builder.getNamedAttr("degree", builder.getI64IntegerAttr(pt.degree)),
          builder.getNamedAttr("is_const", builder.getBoolAttr(pt.isConst)),
      }));
    }
    NamedAttrList finalAbiAttrs;
    finalAbiAttrs.append(
        builder.getNamedAttr("results", builder.getArrayAttr(resultAttrs)));
    finalAbiAttrs.append(
        builder.getNamedAttr("args", builder.getArrayAttr(argAttrs)));
    finalAbiAttrs.append(
        builder.getNamedAttr("pts", builder.getArrayAttr(finalPtAttrs)));
    finalAbiAttrs.append(
        builder.getNamedAttr("keys", builder.getArrayAttr(keyAttrs)));
    if (hasBootstrap) {
      finalAbiAttrs.append(builder.getNamedAttr(
          "boot", builder.getDictionaryAttr({builder.getNamedAttr(
                      "num_slots", builder.getI64IntegerAttr(bootNumSlots))})));
    }
    funcOp->setAttr("cheddar_gpu.abi",
                    finalAbiAttrs.getDictionary(builder.getContext()));

    // Set module-level oracle trace
    if (emitOracleTrace) {
      ModuleOp mod = cast<ModuleOp>(funcOp->getParentOp());
      std::string traceStr = traceOS.str();
      mod->setAttr("cheddar_gpu.oracle_trace", builder.getStringAttr(traceStr));

      // Emit global string constant if not already emitted
      if (!mod.lookupSymbol<memref::GlobalOp>("__cheddar_gpu_oracle_trace")) {
        SmallVector<int8_t> traceBytes;
        for (char c : traceStr) traceBytes.push_back(static_cast<int8_t>(c));
        traceBytes.push_back(0);

        int64_t K = traceBytes.size();
        auto i8Type = builder.getI8Type();
        auto tensorType = RankedTensorType::get({K}, i8Type);
        auto memrefType = MemRefType::get({K}, i8Type);
        auto denseAttr =
            DenseIntElementsAttr::get(tensorType, ArrayRef<int8_t>(traceBytes));

        OpBuilder modBuilder(mod.getBodyRegion());
        memref::GlobalOp::create(
            modBuilder, mod.getLoc(), "__cheddar_gpu_oracle_trace",
            /*sym_visibility=*/modBuilder.getStringAttr("public"),
            /*type=*/memrefType,
            /*initial_value=*/denseAttr,
            /*constant=*/true,
            /*alignment=*/IntegerAttr());
      }
    }

    return success();
  }
};

}  // namespace mlir::heir::lwe
