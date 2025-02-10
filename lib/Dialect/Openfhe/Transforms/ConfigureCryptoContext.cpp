#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"

#include <cstdint>
#include <set>
#include <string>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

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

// Helper function to check if the function has BootstrapOp
bool hasBootstrapOp(func::FuncOp op) {
  bool result = false;
  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<openfhe::BootstrapOp>(op)) {
      result = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

// function that generates the crypto context with proper parameters
LogicalResult generateGenFunc(func::FuncOp op, const std::string &genFuncName,
                              int64_t mulDepth, bool hasBootstrapOp,
                              bool insecure, ImplicitLocOpBuilder &builder) {
  Type openfheContextType =
      openfhe::CryptoContextType::get(builder.getContext());
  SmallVector<Type> funcArgTypes;
  SmallVector<Type> funcResultTypes;
  funcResultTypes.push_back(openfheContextType);

  FunctionType genFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
  auto genFuncOp = builder.create<func::FuncOp>(genFuncName, genFuncType);
  builder.setInsertionPointToEnd(genFuncOp.addEntryBlock());

  // get plaintext modulus from function argument ciphertext type
  // for CKKS, plainMod is 0
  int64_t plainMod = 0;
  for (auto arg : op.getArguments()) {
    if (auto argType = dyn_cast<lwe::NewLWECiphertextType>(
            getElementTypeOrSelf(arg.getType()))) {
      if (auto modArithType = dyn_cast<mod_arith::ModArithType>(
              argType.getPlaintextSpace().getRing().getCoefficientType())) {
        plainMod = modArithType.getModulus().getInt();
        // implicitly assume arguments have the same plaintext modulus
        break;
      }
    }
  }

  // get evalAddCount/KeySwitchCount from func attribute, if present
  int64_t evalAddCount = 0;
  int64_t keySwitchCount = 0;
  if (auto openfheParamsAttr = op->getAttrOfType<mgmt::OpenfheParamsAttr>(
          mgmt::MgmtDialect::kArgOpenfheParamsAttrName)) {
    evalAddCount = openfheParamsAttr.getEvalAddCount();
    keySwitchCount = openfheParamsAttr.getKeySwitchCount();
    // remove the attribute after reading
    op->removeAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName);
  }

  Type openfheParamsType = openfhe::CCParamsType::get(builder.getContext());
  Value ccParams = builder.create<openfhe::GenParamsOp>(
      openfheParamsType, mulDepth, plainMod, insecure, evalAddCount,
      keySwitchCount);
  Value cryptoContext = builder.create<openfhe::GenContextOp>(
      openfheContextType, ccParams,
      BoolAttr::get(builder.getContext(), hasBootstrapOp));

  builder.create<func::ReturnOp>(cryptoContext);
  return success();
}

// function that configures the crypto context with proper keygeneration
LogicalResult generateConfigFunc(
    func::FuncOp op, const std::string &configFuncName, bool hasRelinOp,
    SmallVector<int64_t> rotIndices, bool hasBootstrapOp, int levelBudgetEncode,
    int levelBudgetDecode, ImplicitLocOpBuilder &builder) {
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

  if (hasRelinOp || hasBootstrapOp) {
    builder.create<openfhe::GenMulKeyOp>(cryptoContext, privateKey);
  }
  if (!rotIndices.empty()) {
    builder.create<openfhe::GenRotKeyOp>(cryptoContext, privateKey, rotIndices);
  }
  if (hasBootstrapOp) {
    builder.create<openfhe::SetupBootstrapOp>(
        cryptoContext,
        IntegerAttr::get(IndexType::get(builder.getContext()),
                         levelBudgetEncode),
        IntegerAttr::get(IndexType::get(builder.getContext()),
                         levelBudgetDecode));
    builder.create<openfhe::GenBootstrapKeyOp>(cryptoContext, privateKey);
  }

  builder.create<func::ReturnOp>(cryptoContext);
  return success();
}

LogicalResult convertFunc(func::FuncOp op, int levelBudgetEncode,
                          int levelBudgetDecode, bool insecure) {
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

  bool hasBootstrapOpResult = hasBootstrapOp(op);
  // TODO(#1207): determine mulDepth earlier in mgmt level
  // approxModDepth = 14, this solely depends on secretKeyDist
  // here we use the value for UNIFORM_TERNARY
  int bootstrapDepth = levelBudgetEncode + 14 + levelBudgetDecode;
  if (hasBootstrapOpResult) {
    mulDepth += bootstrapDepth;
  }
  if (failed(generateGenFunc(op, genFuncName, mulDepth, hasBootstrapOpResult,
                             insecure, builder))) {
    return failure();
  }

  builder.setInsertionPointToEnd(module.getBody());

  bool hasRelinOpResult = hasRelinOp(op);
  SmallVector<int64_t> rotIndices = findAllRotIndices(op);
  if (failed(generateConfigFunc(op, configFuncName, hasRelinOpResult,
                                rotIndices, hasBootstrapOpResult,
                                levelBudgetEncode, levelBudgetDecode,
                                builder))) {
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
          if ((funcName == entryFunction) &&
              failed(convertFunc(op, levelBudgetEncode, levelBudgetDecode,
                                 insecure))) {
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
