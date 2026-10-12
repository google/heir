#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Parameters/RLWEParams.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

LogicalResult SchemeParamAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int logN, DenseI64ArrayAttr Q,
    DenseI64ArrayAttr P, int logDefaultScale, CKKSEncryptionType encryptionType,
    CKKSEncryptionTechnique encryptionTechnique,
    DenseI32ArrayAttr moduliPerLevel) {
  if (Q.empty()) {
    return emitError() << "Q must not be empty";
  }
  for (int64_t qi : Q.asArrayRef()) {
    if (qi <= 1) {
      return emitError() << "primes in Q must be > 1, got " << qi;
    }
  }
  if (moduliPerLevel) {
    int64_t totalModuli = 0;
    for (int32_t count : moduliPerLevel.asArrayRef()) {
      if (count <= 0) {
        return emitError() << "moduliPerLevel entries must be positive, got "
                           << count;
      }
      totalModuli += count;
    }
    if (totalModuli != static_cast<int64_t>(Q.size())) {
      return emitError() << "sum of moduliPerLevel (" << totalModuli
                         << ") must match the number of primes in Q ("
                         << Q.size() << ")";
    }
  }
  return success();
}

SmallVector<int64_t> getModuliBitWidths(ArrayRef<int64_t> moduli) {
  SmallVector<int64_t> bitWidths;
  bitWidths.reserve(moduli.size());
  for (int64_t modulus : moduli) {
    bitWidths.push_back(static_cast<int64_t>(std::ceil(std::log2(modulus))));
  }
  return bitWidths;
}

SmallVector<int32_t> getModuliPerLevelOrDefault(SchemeParamAttr attr) {
  if (DenseI32ArrayAttr moduliPerLevel = attr.getModuliPerLevel()) {
    ArrayRef<int32_t> ref = moduliPerLevel.asArrayRef();
    return SmallVector<int32_t>(ref.begin(), ref.end());
  }
  return SmallVector<int32_t>(attr.getQ().size(), 1);
}

SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr) {
  auto logN = attr.getLogN();
  auto ringDim = pow(2, logN);
  auto Q = attr.getQ();
  auto P = attr.getP();
  auto logDefaultScale = attr.getLogDefaultScale();
  ArrayRef<int64_t> qRef = Q.asArrayRef();
  std::vector<int64_t> qiImpl(qRef.begin(), qRef.end());
  std::vector<int64_t> piImpl;
  std::vector<double> logqi;
  std::vector<double> logpi;

  SmallVector<int32_t> moduliPerLevel = getModuliPerLevelOrDefault(attr);
  logqi.reserve(moduliPerLevel.size());
  size_t qIdx = 0;
  for (int32_t count : moduliPerLevel) {
    double levelLogMod = 0.0;
    for (int32_t i = 0; i < count; ++i) {
      levelLogMod += log2(qRef[qIdx++]);
    }
    logqi.push_back(levelLogMod);
  }

  for (auto pi : P.asArrayRef()) {
    piImpl.push_back(pi);
    logpi.push_back(log2(pi));
  }
  auto level = static_cast<int>(logqi.size()) - 1;
  auto dnum = ceil(static_cast<double>(qiImpl.size()) / piImpl.size());
  auto usePublicKey = attr.getEncryptionType() == CKKSEncryptionType::pk;
  auto encryptionTechniqueExtended =
      attr.getEncryptionTechnique() == CKKSEncryptionTechnique::extended;
  return SchemeParam(
      RLWESchemeParam(ringDim, level, logqi, qiImpl, dnum, logpi, piImpl,
                      usePublicKey, encryptionTechniqueExtended),
      logDefaultScale);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
