#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

// Helper function to check if the function has RelinOp
bool hasRelinOp(func::FuncOp op) {
  bool result = false;
  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<openfhe::RelinOp>(op)) {
      result = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

// Helper function to find all the rotation indices in the function
SmallVector<int64_t> findAllRotIndices(func::FuncOp op) {
  std::set<int64_t> distinctRotIndices;
  op.walk([&](openfhe::RotOp rotOp) {
    distinctRotIndices.insert(rotOp.getIndex().getInt());
    return WalkResult::advance();
  });
  SmallVector<int64_t> rotIndicesResult(distinctRotIndices.begin(),
                                        distinctRotIndices.end());
  return rotIndicesResult;
}

// function that generates the crypto context with proper parameters
LogicalResult generateGenFunc(func::FuncOp op, const std::string &genFuncName,
                              int64_t mulDepth, ImplicitLocOpBuilder &builder) {
  Type openfheContextType =
      openfhe::CryptoContextType::get(builder.getContext());
  SmallVector<Type> funcArgTypes;
  SmallVector<Type> funcResultTypes;
  funcResultTypes.push_back(openfheContextType);

  FunctionType genFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
  auto genFuncOp = builder.create<func::FuncOp>(genFuncName, genFuncType);
  builder.setInsertionPointToEnd(genFuncOp.addEntryBlock());

  // TODO(#661) : Calculate the appropriate values by analyzing the function
  int64_t plainMod = 4295294977;
  Type openfheParamsType = openfhe::CCParamsType::get(builder.getContext());
  Value ccParams = builder.create<openfhe::GenParamsOp>(openfheParamsType,
                                                        mulDepth, plainMod);
  Value cryptoContext =
      builder.create<openfhe::GenContextOp>(openfheContextType, ccParams);

  builder.create<func::ReturnOp>(cryptoContext);
  return success();
}

// function that configures the crypto context with proper keygeneration
LogicalResult generateConfigFunc(func::FuncOp op,
                                 const std::string &configFuncName,
                                 bool hasRelinOp,
                                 SmallVector<int64_t> rotIndices,
                                 ImplicitLocOpBuilder &builder) {
  Type openfheContextType =
      openfhe::CryptoContextType::get(builder.getContext());
  Type privateKeyType = openfhe::PrivateKeyType::get(builder.getContext());

  SmallVector<Type> funcArgTypes;
  funcArgTypes.push_back(openfheContextType);
  funcArgTypes.push_back(privateKeyType);

  SmallVector<Type> funcResultTypes;
  funcResultTypes.push_back(openfheContextType);

  FunctionType configFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
  auto configFuncOp =
      builder.create<func::FuncOp>(configFuncName, configFuncType);
  builder.setInsertionPointToEnd(configFuncOp.addEntryBlock());

  Value cryptoContext = configFuncOp.getArgument(0);
  Value privateKey = configFuncOp.getArgument(1);

  if (hasRelinOp) {
    builder.create<openfhe::GenMulKeyOp>(cryptoContext, privateKey);
  }
  if (!rotIndices.empty()) {
    builder.create<openfhe::GenRotKeyOp>(cryptoContext, privateKey, rotIndices);
  }

  builder.create<func::ReturnOp>(cryptoContext);
  return success();
}

LogicalResult convertFunc(func::FuncOp op) {
  auto module = op->getParentOfType<ModuleOp>();
  std::string genFuncName("");
  llvm::raw_string_ostream genNameOs(genFuncName);
  genNameOs << op.getSymName() << "__generate_crypto_context";

  std::string configFuncName("");
  llvm::raw_string_ostream configNameOs(configFuncName);
  configNameOs << op.getSymName() << "__configure_crypto_context";

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  // get mulDepth from function argument ciphertext type
  int64_t mulDepth = 0;
  for (auto arg : op.getArguments()) {
    if (auto argType = dyn_cast<lwe::NewLWECiphertextType>(
            getElementTypeOrSelf(arg.getType()))) {
      if (auto rnsType = dyn_cast<rns::RNSType>(
              argType.getCiphertextSpace().getRing().getCoefficientType())) {
        mulDepth = rnsType.getBasisTypes().size() - 1;
        // implicitly assume arguments have the same level
        break;
      }
    }
  }

  if (failed(generateGenFunc(op, genFuncName, mulDepth, builder))) {
    return failure();
  }

  builder.setInsertionPointToEnd(module.getBody());

  bool hasRelinOpResult = hasRelinOp(op);
  SmallVector<int64_t> rotIndices = findAllRotIndices(op);
  if (failed(generateConfigFunc(op, configFuncName, hasRelinOpResult,
                                rotIndices, builder))) {
    return failure();
  }
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
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
