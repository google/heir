#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"

#include <cstdint>
#include <set>
#include <string>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoAttributes.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/TransformUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

// Helper function to check if the function has RelinearizeOp
bool hasRelinOp(func::FuncOp op) {
  bool result = false;
  walkFuncAndCallees(op, [&](Operation* op) {
    if (isa<BGVRelinearizeOp, BGVRelinearizeNewOp, CKKSRelinearizeOp,
            CKKSRelinearizeNewOp>(op)) {
      result = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

// Helper function to check if the function has BootstrapOp
bool hasBootstrapOp(func::FuncOp funcOp) {
  auto result = walkFuncAndCallees(funcOp, [&](Operation* op) {
    if (isa<lattigo::CKKSBootstrapOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Helper function to find all the rotation indices in the function
// TODO(#1186): handle rotate rows
SmallVector<int64_t> findAllRotIndices(func::FuncOp op) {
  std::set<int64_t> distinctRotIndices;
  walkFuncAndCallees(op, [&](Operation* op) {
    llvm::TypeSwitch<Operation*>(op)
        .Case<BGVRotateColumnsNewOp>([&](BGVRotateColumnsNewOp rotOp) {
          distinctRotIndices.insert(rotOp.getOffset().getInt());
          return WalkResult::advance();
        })
        .Case<BGVRotateColumnsOp>([&](BGVRotateColumnsOp rotOp) {
          distinctRotIndices.insert(rotOp.getOffset().getInt());
          return WalkResult::advance();
        })
        .Case<CKKSRotateOp>([&](CKKSRotateOp rotOp) {
          distinctRotIndices.insert(rotOp.getOffset().getInt());
          return WalkResult::advance();
        })
        .Case<CKKSRotateNewOp>([&](CKKSRotateNewOp rotOp) {
          distinctRotIndices.insert(rotOp.getOffset().getInt());
          return WalkResult::advance();
        })
        .Default([&](Operation* op) { return WalkResult::advance(); });
    return WalkResult::advance();
  });
  SmallVector<int64_t> rotIndicesResult(distinctRotIndices.begin(),
                                        distinctRotIndices.end());
  return rotIndicesResult;
}

// Helper function to find encryptor type used in the whole module
// assume only one possible type on encryptor in the whole module
RLWEEncryptorType findEncryptorType(ModuleOp module) {
  Type ret = nullptr;
  module->walk([&](func::FuncOp funcOp) {
    // also work for funcOp declaration
    for (auto argType : funcOp.getArgumentTypes()) {
      if (mlir::isa<RLWEEncryptorType>(argType)) {
        ret = argType;
      }
    }
  });
  if (ret) {
    return mlir::cast<RLWEEncryptorType>(ret);
  }
  // default to public key encryption
  return RLWEEncryptorType::get(module.getContext(), /*publicKey*/ true);
}

template <bool IsBFV>
struct LattigoBGVScheme {
  using EvaluatorType = BGVEvaluatorType;
  using ParameterType = BGVParameterType;
  using EncoderType = BGVEncoderType;
  using ParametersLiteralAttrType = BGVParametersLiteralAttr;
  using NewParametersFromLiteralOp = BGVNewParametersFromLiteralOp;
  using NewEncoderOp = BGVNewEncoderOp;
  using NewEvaluatorOp = BGVNewEvaluatorOp;
  using SchemeParamAttrType = bgv::SchemeParamAttr;

  static int getLogN(Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      return schemeParamAttr.getLogN();
    }
    // default logN
    return 14;
  }

  static ParametersLiteralAttrType getParametersLiteralAttr(
      MLIRContext* ctx, Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      auto logN = schemeParamAttr.getLogN();
      auto Q = schemeParamAttr.getQ();
      auto P = schemeParamAttr.getP();
      auto ptm = schemeParamAttr.getPlaintextModulus();
      return ParametersLiteralAttrType::get(ctx, logN, Q, P,
                                            /*logQ*/ nullptr, /*logP*/ nullptr,
                                            ptm);
    }
    // default parameters
    // 128-bit secure parameters enabling depth-7 circuits.
    // LogN:14, LogQP: 431.
    return ParametersLiteralAttrType::get(
        ctx, /*logN*/ getLogN(moduleOp), /*Q*/ nullptr, /*P*/ nullptr,
        /*logQ*/
        DenseI32ArrayAttr::get(ctx, {55, 45, 45, 45, 45, 45, 45, 45}),
        /*logP*/ DenseI32ArrayAttr::get(ctx, {61}),
        /*ptm*/ 0x10001);
  }

  static SchemeParamAttrType getSchemeParamAttr(Operation* moduleOp) {
    return moduleOp->getAttrOfType<bgv::SchemeParamAttr>(
        bgv::BGVDialect::kSchemeParamAttrName);
  }

  static void cleanSchemeParamAttr(Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      moduleOp->removeAttr(bgv::BGVDialect::kSchemeParamAttrName);
    }
  }

  static Value getNewEvaluatorOp(ImplicitLocOpBuilder& builder, Value params,
                                 Value evalKeySet) {
    return NewEvaluatorOp::create(builder,
                                  EvaluatorType::get(builder.getContext()),
                                  params, evalKeySet, IsBFV);
  }
};

struct LattigoCKKSScheme {
  using EvaluatorType = CKKSEvaluatorType;
  using BootstrappingEvaluatorType = CKKSBootstrappingEvaluatorType;
  using ParameterType = CKKSParameterType;
  using EncoderType = CKKSEncoderType;
  using ParametersLiteralAttrType = CKKSParametersLiteralAttr;
  using NewParametersFromLiteralOp = CKKSNewParametersFromLiteralOp;
  using NewEncoderOp = CKKSNewEncoderOp;
  using NewEvaluatorOp = CKKSNewEvaluatorOp;
  using SchemeParamAttrType = ckks::SchemeParamAttr;

  static int getLogN(Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      return schemeParamAttr.getLogN();
    }
    // default logN
    return 14;
  }

  static ParametersLiteralAttrType getParametersLiteralAttr(
      MLIRContext* ctx, Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      auto logN = schemeParamAttr.getLogN();
      auto Q = schemeParamAttr.getQ();
      auto P = schemeParamAttr.getP();
      auto logDefaultScale = schemeParamAttr.getLogDefaultScale();
      return ParametersLiteralAttrType::get(ctx, logN, Q, P,
                                            /*logQ*/ nullptr, /*logP*/ nullptr,
                                            logDefaultScale);
    }
    // 128-bit secure parameters enabling depth-7 circuits.
    // LogN:14, LogQP: 431.
    return ParametersLiteralAttrType::get(
        ctx, /*logN*/ getLogN(moduleOp), /*Q*/ nullptr,
        /*P*/ nullptr,
        /*logQ*/
        DenseI32ArrayAttr::get(ctx, {55, 45, 45, 45, 45, 45, 45, 45}),
        /*logP*/ DenseI32ArrayAttr::get(ctx, {61}),
        /*logDefaultScale*/ 45);
  }

  static CKKSBootstrappingParametersLiteralAttr
  getBootstrappingParametersLiteralAttr(MLIRContext* ctx, Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      auto logN = schemeParamAttr.getLogN();
      return CKKSBootstrappingParametersLiteralAttr::get(ctx, logN);
    }
    return CKKSBootstrappingParametersLiteralAttr::get(
        ctx, /*logN*/ getLogN(moduleOp));
  }

  static SchemeParamAttrType getSchemeParamAttr(Operation* moduleOp) {
    return moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
  }

  static void cleanSchemeParamAttr(Operation* moduleOp) {
    auto schemeParamAttr = getSchemeParamAttr(moduleOp);
    if (schemeParamAttr) {
      moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }
  }

  static Value getNewEvaluatorOp(ImplicitLocOpBuilder& builder, Value params,
                                 Value evalKeySet) {
    return NewEvaluatorOp::create(
        builder, EvaluatorType::get(builder.getContext()), params, evalKeySet);
  }
};

template <typename LattigoScheme>
LogicalResult convertFuncForScheme(func::FuncOp op) {
  using EvaluatorType = typename LattigoScheme::EvaluatorType;
  using ParameterType = typename LattigoScheme::ParameterType;
  using EncoderType = typename LattigoScheme::EncoderType;
  using NewParametersFromLiteralOp =
      typename LattigoScheme::NewParametersFromLiteralOp;
  using NewEncoderOp = typename LattigoScheme::NewEncoderOp;

  auto module = op->getParentOfType<ModuleOp>();
  std::string configFuncName("");
  llvm::raw_string_ostream configNameOs(configFuncName);
  configNameOs << op.getSymName() << "__configure";

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  // __configure() -> ((optional) bootstrapping evaluator, evaluator, params,
  // encoder, encryptor, decryptor)
  SmallVector<Type> funcArgTypes;
  SmallVector<Type> funcResultTypes;

  Type bootstrappingEvaluatorType =
      CKKSBootstrappingEvaluatorType::get(builder.getContext());
  Type evaluatorType = EvaluatorType::get(builder.getContext());
  Type paramsType = ParameterType::get(builder.getContext());
  Type encoderType = EncoderType::get(builder.getContext());
  RLWEEncryptorType encryptorType = findEncryptorType(module);
  Type decryptorType = RLWEDecryptorType::get(builder.getContext());

  bool hasBootstrap = hasBootstrapOp(op);
  if (hasBootstrap) {
    funcResultTypes.push_back(bootstrappingEvaluatorType);
  }
  funcResultTypes.push_back(evaluatorType);
  funcResultTypes.push_back(paramsType);
  funcResultTypes.push_back(encoderType);
  funcResultTypes.push_back(encryptorType);
  funcResultTypes.push_back(decryptorType);

  FunctionType configFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);

  auto configFuncOp =
      func::FuncOp::create(builder, configFuncName, configFuncType);
  builder.setInsertionPointToEnd(configFuncOp.addEntryBlock());

  auto* moduleOp = op->getParentOp();
  int logN = LattigoScheme::getLogN(moduleOp);
  auto paramAttr =
      LattigoScheme::getParametersLiteralAttr(builder.getContext(), moduleOp);
  LattigoScheme::cleanSchemeParamAttr(moduleOp);

  auto paramType = ParameterType::get(builder.getContext());
  auto params =
      NewParametersFromLiteralOp::create(builder, paramType, paramAttr);

  auto encoder = NewEncoderOp::create(builder, encoderType, params);
  auto kgenType = RLWEKeyGeneratorType::get(builder.getContext());
  auto kgen = RLWENewKeyGeneratorOp::create(builder, kgenType, params);
  auto skType = RLWESecretKeyType::get(builder.getContext());
  auto pkType = RLWEPublicKeyType::get(builder.getContext());
  SmallVector<Type> keypairTypes = {skType, pkType};
  auto keypair = RLWEGenKeyPairOp::create(builder, keypairTypes, kgen);
  auto sk = keypair.getResult(0);
  auto pk = keypair.getResult(1);
  auto encKey = encryptorType.getPublicKey() ? pk : sk;
  auto encryptor =
      RLWENewEncryptorOp::create(builder, encryptorType, params, encKey);
  auto decryptor =
      RLWENewDecryptorOp::create(builder, decryptorType, params, sk);

  SmallVector<Value> evalKeys;

  // generate Relinearization Key on demand
  if (hasRelinOp(op)) {
    auto rkType = RLWERelinearizationKeyType::get(builder.getContext());
    auto rk = RLWEGenRelinearizationKeyOp::create(builder, rkType, kgen, sk);
    evalKeys.push_back(rk);
  }

  // generate Galois Keys on demand
  auto rotIndices = findAllRotIndices(op);
  for (auto rotIndex : rotIndices) {
    auto galoisElement = 1;
    while (rotIndex > 0) {
      galoisElement = (galoisElement * 5) % (1 << (logN + 1));
      rotIndex--;
    }
    auto galoisElementAttr = IntegerAttr::get(
        IntegerType::get(builder.getContext(), 64), galoisElement);
    auto gkType =
        RLWEGaloisKeyType::get(builder.getContext(), galoisElementAttr);
    auto gk = RLWEGenGaloisKeyOp::create(builder, gkType, kgen, sk,
                                         galoisElementAttr);
    evalKeys.push_back(gk);
  }

  Value evalKeySet(nullptr);
  if (!evalKeys.empty()) {
    auto evalKeyType = RLWEEvaluationKeySetType::get(builder.getContext());
    evalKeySet =
        RLWENewEvaluationKeySetOp::create(builder, evalKeyType, evalKeys);
  }

  // evalKeySet is optional so nulltpr is acceptable
  SmallVector<Value> results;
  if (hasBootstrap) {
    auto ctx = builder.getContext();
    auto btParamsLiteral =
        LattigoCKKSScheme::getBootstrappingParametersLiteralAttr(ctx, moduleOp);
    auto btParams = CKKSNewBootstrappingParametersFromLiteralOp::create(
        builder, params, btParamsLiteral);
    auto bEvalKeySet = lattigo::CKKSGenEvaluationKeysBootstrappingOp::create(
        builder, btParams, sk);
    auto bEval = lattigo::CKKSNewBootstrappingEvaluatorOp::create(
        builder, btParams, bEvalKeySet);
    results.push_back(bEval);
  }
  Value evaluator =
      LattigoScheme::getNewEvaluatorOp(builder, params, evalKeySet);

  results.append({evaluator, params, encoder, encryptor, decryptor});
  func::ReturnOp::create(builder, results);
  return success();
}

LogicalResult convertFunc(func::FuncOp op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (moduleIsBGV(module)) {
    return convertFuncForScheme<LattigoBGVScheme</*IsBFV*/ false>>(op);
  }
  if (moduleIsBFV(module)) {
    return convertFuncForScheme<LattigoBGVScheme</*IsBFV*/ true>>(op);
  }
  if (moduleIsCKKS(module)) {
    return convertFuncForScheme<LattigoCKKSScheme>(op);
  }
  return op->emitError("Unknown scheme");
}

struct ConfigureCryptoContext
    : impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

  void runOnOperation() override {
    auto funcOp =
        detectEntryFunction(cast<ModuleOp>(getOperation()), entryFunction);
    if (funcOp && failed(convertFunc(funcOp))) {
      funcOp->emitError("Failed to configure the crypto context for func");
      signalPassFailure();
    }
  }
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
