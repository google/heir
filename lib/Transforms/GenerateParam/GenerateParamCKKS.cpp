#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project

#define DEBUG_TYPE "GenerateParamCKKS"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMCKKS
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamCKKS : impl::GenerateParamCKKSBase<GenerateParamCKKS> {
  using GenerateParamCKKSBase::GenerateParamCKKSBase;

  // assume only one main func
  // also assume max level at entry
  // also assume first genericOp arg is secret
  int getMaxLevel() {
    int maxLevel = 0;
    getOperation()->walk([&](func::FuncOp funcOp) {
      funcOp->walk([&](secret::GenericOp genericOp) {
        if (genericOp.getBody()->getNumArguments() > 0) {
          maxLevel = getLevelFromMgmtAttr(genericOp.getBody()->getArgument(0));
        }
      });
    });
    return maxLevel;
  }

  void runOnOperation() override {
    auto maxLevel = getMaxLevel();

    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                ckks::CKKSDialect::kSchemeParamAttrName)) {
      // TODO: put this in validate-noise once CKKS noise model is in
      auto schemeParam =
          ckks::SchemeParam::getSchemeParamFromAttr(schemeParamAttr);
      if (schemeParam.getLevel() < maxLevel) {
        getOperation()->emitOpError()
            << "The level in the scheme param is smaller than the max level.\n";
        signalPassFailure();
        return;
      }
      return;
    }

    // generate scheme parameters
    std::vector<double> logPrimes(maxLevel + 1, scalingModBits);
    logPrimes[0] = firstModBits;

    auto schemeParam = ckks::SchemeParam::getConcreteSchemeParam(
        logPrimes, scalingModBits, slotNumber, usePublicKey);

    LLVM_DEBUG(llvm::dbgs() << "Scheme Param:\n" << schemeParam << "\n");

    // annotate ckks::SchemeParamAttr to ModuleOp
    getOperation()->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            &getContext(), log2(schemeParam.getRingDim()),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getPi())),
            schemeParam.getLogDefaultScale(),
            usePublicKey ? ckks::CKKSEncryptionType::pk
                         : ckks::CKKSEncryptionType::sk));
  }
};

}  // namespace heir
}  // namespace mlir
