#include "lib/Dialect/JaxiteWord/Transforms/JaxiteCkksParameterSelection.h"

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordAttributes.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/ADT/APInt.h"           // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

#define GEN_PASS_DEF_JAXITECKKSPARAMETERSELECTION
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

struct JaxiteCkksParameterSelection
    : impl::JaxiteCkksParameterSelectionBase<JaxiteCkksParameterSelection> {
  using JaxiteCkksParameterSelectionBase::JaxiteCkksParameterSelectionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      module->emitOpError() << "Missing ckks.schemeParam attribute";
      signalPassFailure();
      return;
    }

    int logN = schemeParamAttr.getLogN();
    int ringDim = 1 << logN;

    auto Q = schemeParamAttr.getQ().asArrayRef();
    auto P = schemeParamAttr.getP().asArrayRef();

    int totalBitsQ = 0;
    for (auto q : Q) {
      totalBitsQ += llvm::APInt(64, q).getActiveBits();
    }

    int totalBitsP = 0;
    for (auto p : P) {
      totalBitsP += llvm::APInt(64, p).getActiveBits();
    }

    std::vector<int64_t> existingPrimes;
    std::vector<int64_t> qTowers;
    std::vector<int64_t> pTowers;

    int bitsGeneratedQ = 0;
    while (bitsGeneratedQ < totalBitsQ) {
      int64_t prime = findPrime(30, ringDim, existingPrimes);
      qTowers.push_back(prime);
      existingPrimes.push_back(prime);
      bitsGeneratedQ += 30;
    }

    int bitsGeneratedP = 0;
    while (bitsGeneratedP < totalBitsP) {
      int64_t prime = findPrime(30, ringDim, existingPrimes);
      pTowers.push_back(prime);
      existingPrimes.push_back(prime);
      bitsGeneratedP += 30;
    }

    auto qTowersAttr = DenseI64ArrayAttr::get(context, qTowers);
    auto pTowersAttr = DenseI64ArrayAttr::get(context, pTowers);

    int dnum = computeDnum(Q.size() - 1);

    // FIXME: Replace dummy value for composite_degree.
    auto ckksParamsAttr = CkksParametersAttr::get(
        context, qTowersAttr, pTowersAttr, 4, 4, dnum, 7, 1);

    module->setAttr("jaxiteword.ckks_params", ckksParamsAttr);
  }
};

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
