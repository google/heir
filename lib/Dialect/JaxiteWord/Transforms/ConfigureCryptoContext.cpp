#include "lib/Dialect/JaxiteWord/Transforms/ConfigureCryptoContext.h"

#include <cmath>
#include <cstdint>
#include <set>
#include <string>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/TransformUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

struct Config {
  SmallVector<int64_t> rotIndices;
  int64_t degree;
  int64_t numSlots;
  double scalingFactor;
  SmallVector<int64_t> qTowers;
  SmallVector<int64_t> pTowers;
  int dnum;
  int r;
  int c;
  int batch;
};

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

struct ConfigureCryptoContext
    : impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

 private:
  Config config;

  SmallVector<int64_t> findAllRotIndices(func::FuncOp op) {
    std::set<int64_t> distinctRotIndices;
    walkFuncAndCallees(op, [&](Operation* op) {
      if (auto rotOp = dyn_cast<jaxiteword::RotOp>(op)) {
        distinctRotIndices.insert(rotOp.getIndex());
      }
      return WalkResult::advance();
    });
    return SmallVector<int64_t>(distinctRotIndices.begin(),
                                distinctRotIndices.end());
  }

  LogicalResult generateGenFunc(func::FuncOp op, const std::string& genFuncName,
                                ImplicitLocOpBuilder& builder) {
    Type ccType = CryptoContextType::get(builder.getContext());

    SmallVector<Type> funcArgTypes;
    SmallVector<Type> funcResultTypes = {ccType};

    FunctionType genFuncType =
        FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
    auto genFuncOp = func::FuncOp::create(builder, genFuncName, genFuncType);
    builder.setInsertionPointToEnd(genFuncOp.addEntryBlock());

    Value cryptoContext = GenParamsOp::create(
        builder, ccType,
        /*degree=*/static_cast<uint64_t>(config.degree),
        /*numSlots=*/static_cast<uint64_t>(config.numSlots),
        /*scalingFactor=*/llvm::APFloat(config.scalingFactor),
        /*qTowers=*/ArrayRef<int64_t>(config.qTowers),
        /*pTowers=*/ArrayRef<int64_t>(config.pTowers),
        /*batch=*/static_cast<uint32_t>(config.batch),
        /*r=*/static_cast<uint32_t>(config.r),
        /*c=*/static_cast<uint32_t>(config.c),
        /*dnum=*/static_cast<uint32_t>(config.dnum),
        /*compositeDegree=*/static_cast<uint32_t>(compositeDegree));

    func::ReturnOp::create(builder, cryptoContext);
    return success();
  }

  LogicalResult generateConfigFunc(func::FuncOp op,
                                   const std::string& configFuncName,
                                   ImplicitLocOpBuilder& builder) {
    Type ccType = CryptoContextType::get(builder.getContext());
    Type pkType = PublicKeyType::get(builder.getContext());
    Type skType = PrivateKeyType::get(builder.getContext());
    Type ekType = EvalKeyType::get(builder.getContext());

    SmallVector<Type> funcArgTypes = {ccType, pkType, skType, ekType};
    SmallVector<Type> funcResultTypes;

    FunctionType configFuncType =
        FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
    auto configFuncOp =
        func::FuncOp::create(builder, configFuncName, configFuncType);
    builder.setInsertionPointToEnd(configFuncOp.addEntryBlock());

    Value cryptoContext = configFuncOp.getArgument(0);
    Value publicKey = configFuncOp.getArgument(1);
    Value secretKey = configFuncOp.getArgument(2);
    Value evaluationKey = configFuncOp.getArgument(3);

    ProgramInitializationOp::create(builder, cryptoContext, publicKey,
                                    secretKey, evaluationKey,
                                    /*totalRotationIndices=*/config.rotIndices,
                                    /*dnum=*/config.dnum,
                                    /*r=*/config.r,
                                    /*c=*/config.c,
                                    /*batch=*/config.batch);

    func::ReturnOp::create(builder, ValueRange{});
    return success();
  }

  LogicalResult convertFunc(func::FuncOp op) {
    auto module = op->getParentOfType<ModuleOp>();
    std::string genFuncName;
    llvm::raw_string_ostream genNameOs(genFuncName);
    genNameOs << op.getSymName() << "__generate_crypto_context";

    std::string configFuncName;
    llvm::raw_string_ostream configNameOs(configFuncName);
    configNameOs << op.getSymName() << "__configure_crypto_context";

    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

    if (failed(generateGenFunc(op, genFuncName, builder))) {
      return failure();
    }

    builder.setInsertionPointToEnd(module.getBody());

    if (failed(generateConfigFunc(op, configFuncName, builder))) {
      return failure();
    }
    return success();
  }

  LogicalResult getConfig(func::FuncOp op) {
    auto module = op->getParentOfType<ModuleOp>();

    config.degree = 0;
    config.numSlots = 0;
    config.scalingFactor = scalingFactor;
    config.qTowers = {};
    config.pTowers = {};

    if (auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
            ckks::CKKSDialect::kSchemeParamAttrName)) {
      int logN = schemeParamAttr.getLogN();
      config.degree = 1 << logN;
      config.numSlots = 1 << (logN - 1);
      config.scalingFactor =
          std::pow(2.0, schemeParamAttr.getLogDefaultScale());

      auto qArr = schemeParamAttr.getQ().asArrayRef();
      config.qTowers.assign(qArr.begin(), qArr.end());

      auto pArr = schemeParamAttr.getP().asArrayRef();
      config.pTowers.assign(pArr.begin(), pArr.end());

      module->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    config.rotIndices = findAllRotIndices(op);

    config.dnum = dnum;
    config.batch = batch;

    if (r != 0 && c != 0) {
      config.r = r;
      config.c = c;
    } else if (config.degree > 0) {
      int logN = static_cast<int>(std::log2(config.degree));
      config.r = 1 << (logN / 2);
      config.c = config.degree / config.r;
    } else {
      config.r = 0;
      config.c = 0;
    }

    return success();
  }

 public:
  void runOnOperation() override {
    auto funcOp =
        detectEntryFunction(cast<ModuleOp>(getOperation()), entryFunction);
    if (funcOp && succeeded(getConfig(funcOp)) && failed(convertFunc(funcOp))) {
      funcOp->emitError("Failed to configure the crypto context for func");
      signalPassFailure();
    }
  }
};

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
