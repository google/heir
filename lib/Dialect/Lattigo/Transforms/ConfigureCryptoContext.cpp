#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

// Helper function to check if the function has RelinearizeOp
bool hasRelinOp(func::FuncOp op) {
  bool result = false;
  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<BGVRelinearizeOp>(op)) {
      result = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

// Helper function to find all the rotation indices in the function
// TODO(#1186): handle rotate rows
SmallVector<int64_t> findAllRotIndices(func::FuncOp op) {
  std::set<int64_t> distinctRotIndices;
  op.walk([&](BGVRotateColumnsOp rotOp) {
    distinctRotIndices.insert(rotOp.getOffset().getInt());
    return WalkResult::advance();
  });
  SmallVector<int64_t> rotIndicesResult(distinctRotIndices.begin(),
                                        distinctRotIndices.end());
  return rotIndicesResult;
}

LogicalResult convertFunc(func::FuncOp op) {
  auto module = op->getParentOfType<ModuleOp>();
  std::string configFuncName("");
  llvm::raw_string_ostream configNameOs(configFuncName);
  configNameOs << op.getSymName() << "__configure";

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  // __configure() -> (evaluator, params, encoder, encryptor, decryptor)
  SmallVector<Type> funcArgTypes;
  SmallVector<Type> funcResultTypes;

  Type evaluatorType = BGVEvaluatorType::get(builder.getContext());
  Type paramsType = BGVParameterType::get(builder.getContext());
  Type encoderType = BGVEncoderType::get(builder.getContext());
  Type encryptorType = RLWEEncryptorType::get(builder.getContext());
  Type decryptorType = RLWEDecryptorType::get(builder.getContext());
  funcResultTypes.push_back(evaluatorType);
  funcResultTypes.push_back(paramsType);
  funcResultTypes.push_back(encoderType);
  funcResultTypes.push_back(encryptorType);
  funcResultTypes.push_back(decryptorType);

  FunctionType configFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);

  auto configFuncOp =
      builder.create<func::FuncOp>(configFuncName, configFuncType);
  builder.setInsertionPointToEnd(configFuncOp.addEntryBlock());

  // TODO(#465): Allow custom params
  // 128-bit secure parameters enabling depth-7 circuits.
  // LogN:14, LogQP: 431.
  auto logN = 14;
  auto paramAttr = BGVParametersLiteralAttr::get(
      builder.getContext(), /*logN*/ logN, /*Q*/ nullptr, /*P*/ nullptr,
      /*logQ*/
      DenseI32ArrayAttr::get(builder.getContext(),
                             {55, 45, 45, 45, 45, 45, 45, 45}),
      /*logP*/ DenseI32ArrayAttr::get(builder.getContext(), {61}),
      /*ptm*/ 0x10001);
  auto paramType = BGVParameterType::get(builder.getContext());

  auto params =
      builder.create<BGVNewParametersFromLiteralOp>(paramType, paramAttr);

  auto encoder = builder.create<BGVNewEncoderOp>(encoderType, params);
  auto kgenType = RLWEKeyGeneratorType::get(builder.getContext());
  auto kgen = builder.create<RLWENewKeyGeneratorOp>(kgenType, params);
  auto skType = RLWESecretKeyType::get(builder.getContext());
  auto pkType = RLWEPublicKeyType::get(builder.getContext());
  SmallVector<Type> keypairTypes = {skType, pkType};
  auto keypair = builder.create<RLWEGenKeyPairOp>(keypairTypes, kgen);
  auto sk = keypair.getResult(0);
  auto pk = keypair.getResult(1);
  auto encryptor =
      builder.create<RLWENewEncryptorOp>(encryptorType, params, pk);
  auto decryptor =
      builder.create<RLWENewDecryptorOp>(decryptorType, params, sk);

  SmallVector<Value> evalKeys;

  // generate Relinearization Key on demand
  if (hasRelinOp(op)) {
    auto rkType = RLWERelinearizationKeyType::get(builder.getContext());
    auto rk = builder.create<RLWEGenRelinearizationKeyOp>(rkType, kgen, sk);
    evalKeys.push_back(rk);
  }

  // generate Galois Keys on demand
  auto rotIndices = findAllRotIndices(op);
  for (auto rotIndex : rotIndices) {
    auto galoisElement = int(pow(5, rotIndex)) % (1 << logN);
    auto galoisElementAttr = IntegerAttr::get(
        IntegerType::get(builder.getContext(), 64), galoisElement);
    auto gkType =
        RLWEGaloisKeyType::get(builder.getContext(), galoisElementAttr);
    auto gk =
        builder.create<RLWEGenGaloisKeyOp>(gkType, kgen, sk, galoisElementAttr);
    evalKeys.push_back(gk);
  }

  Value evalKeySet(nullptr);
  if (!evalKeys.empty()) {
    auto evalKeyType = RLWEEvaluationKeySetType::get(builder.getContext());
    evalKeySet =
        builder.create<RLWENewEvaluationKeySetOp>(evalKeyType, evalKeys);
  }

  // evalKeySet is optional so nulltpr is acceptable
  auto evaluator =
      builder.create<BGVNewEvaluatorOp>(evaluatorType, params, evalKeySet);

  SmallVector<Value> results = {evaluator, params, encoder, encryptor,
                                decryptor};
  builder.create<func::ReturnOp>(results);
  return success();
}

struct ConfigureCryptoContext
    : impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

  void runOnOperation() override {
    auto result =
        getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
          auto funcName = op.getSymName();
          if ((funcName == entryFunction) && failed(convertFunc(op))) {
            op->emitError("Failed to configure the crypto context for func");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
