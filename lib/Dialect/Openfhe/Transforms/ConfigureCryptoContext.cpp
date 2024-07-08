#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>

#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
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

// Helper function to check if the function has MulOp or MulNoRelinOp
bool hasMulOp(func::FuncOp op) {
  bool result = false;
  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<openfhe::MulOp>(op) || isa<openfhe::MulNoRelinOp>(op) ||
        isa<openfhe::MulPlainOp>(op)) {
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
                                 bool hasMulOp, SmallVector<int64_t> rotIndices,
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

  if (hasMulOp) {
    builder.create<openfhe::GenMulKeyOp>(cryptoContext, privateKey);
  }
  if (!rotIndices.empty()) {
    builder.create<openfhe::GenRotKeyOp>(cryptoContext, privateKey, rotIndices);
  }

  builder.create<func::ReturnOp>(cryptoContext);
  return success();
}

LogicalResult convertFunc(func::FuncOp op, int64_t mulDepth) {
  auto module = op->getParentOfType<ModuleOp>();
  std::string genFuncName("");
  llvm::raw_string_ostream genNameOs(genFuncName);
  genNameOs << op.getSymName() << "__generate_crypto_context";

  std::string configFuncName("");
  llvm::raw_string_ostream configNameOs(configFuncName);
  configNameOs << op.getSymName() << "__configure_crypto_context";

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  if (failed(generateGenFunc(op, genFuncName, mulDepth, builder))) {
    return failure();
  }

  builder.setInsertionPointToEnd(module.getBody());

  bool hasMulOpResult = hasMulOp(op);
  SmallVector<int64_t> rotIndices = findAllRotIndices(op);
  if (failed(generateConfigFunc(op, configFuncName, hasMulOpResult, rotIndices,
                                builder))) {
    return failure();
  }
  return success();
}

struct ConfigureCryptoContext
    : impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

  void runOnOperation() override {
    // Analyse the operations to find the MulDepth
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<MulDepthAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }
    int64_t maxMulDepth = 0;
    // walk the operations to find the max MulDepth
    getOperation()->walk([&](Operation *op) {
      // if the lengths of the operands is 0, then return
      if (op->getNumResults() == 0) return WalkResult::advance();
      const MulDepthLattice *resultLattice =
          solver.lookupState<MulDepthLattice>(op->getResult(0));
      if (resultLattice->getValue().isInitialized()) {
        maxMulDepth =
            std::max(maxMulDepth, resultLattice->getValue().getValue());
      }
      return WalkResult::advance();
    });

    auto result =
        getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
          auto funcName = op.getSymName();
          if ((funcName == entryFunction) &&
              failed(convertFunc(op, maxMulDepth))) {
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
