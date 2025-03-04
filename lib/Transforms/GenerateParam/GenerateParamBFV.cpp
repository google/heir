#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project

#define DEBUG_TYPE "GenerateParamBFV"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMBFV
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

struct GenerateParamBFV : impl::GenerateParamBFVBase<GenerateParamBFV> {
  using GenerateParamBFVBase::GenerateParamBFVBase;

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

  void annotateSchemeParam(const bgv::SchemeParam &schemeParam) {
    getOperation()->setAttr(
        bgv::BGVDialect::kSchemeParamAttrName,
        bgv::SchemeParamAttr::get(
            &getContext(), log2(schemeParam.getRingDim()),

            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(&getContext(),
                                   ArrayRef(schemeParam.getPi())),
            schemeParam.getPlaintextModulus()));
  }

  void generateFallbackParam() {
    // generate fallback scheme parameters
    auto maxLevel = getMaxLevel();
    std::vector<double> logPrimes(maxLevel + 1, 60);  // all primes of 60 bits

    auto schemeParam = bgv::SchemeParam::getConcreteSchemeParam(
        logPrimes, plaintextModulus, slotNumber);

    annotateSchemeParam(schemeParam);
  }

  void runOnOperation() override {
    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
                bgv::BGVDialect::kSchemeParamAttrName)) {
      getOperation()->emitRemark()
          << "Scheme parameters already exist. Skipping generation.\n";
      return;
    }

    generateFallbackParam();
  }
};

}  // namespace heir
}  // namespace mlir
