#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"   // from @llvm-project
#include "llvm/include/llvm/Support/SourceMgr.h"      // from @llvm-project
#include "llvm/include/llvm/Support/YAMLTraits.h"     // from @llvm-project
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

struct YamlCKKSSchemeParam {
  int logN = 0;
  std::vector<int64_t> Q;
  std::vector<int64_t> P;
  int logDefaultScale = 0;
  mlir::heir::ckks::CKKSEncryptionType encryptionType =
      mlir::heir::ckks::CKKSEncryptionType::pk;
  mlir::heir::ckks::CKKSEncryptionTechnique encryptionTechnique =
      mlir::heir::ckks::CKKSEncryptionTechnique::standard;
  std::vector<int32_t> moduliPerLevel;
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

namespace llvm {
namespace yaml {

template <>
struct ScalarEnumerationTraits<mlir::heir::ckks::CKKSEncryptionType> {
  static void enumeration(IO& io, mlir::heir::ckks::CKKSEncryptionType& val) {
    io.enumCase(val, "pk", mlir::heir::ckks::CKKSEncryptionType::pk);
    io.enumCase(val, "sk", mlir::heir::ckks::CKKSEncryptionType::sk);
  }
};

template <>
struct ScalarEnumerationTraits<mlir::heir::ckks::CKKSEncryptionTechnique> {
  static void enumeration(IO& io,
                          mlir::heir::ckks::CKKSEncryptionTechnique& val) {
    io.enumCase(val, "standard",
                mlir::heir::ckks::CKKSEncryptionTechnique::standard);
    io.enumCase(val, "extended",
                mlir::heir::ckks::CKKSEncryptionTechnique::extended);
  }
};

template <>
struct MappingTraits<mlir::heir::ckks::YamlCKKSSchemeParam> {
  static void mapping(IO& io, mlir::heir::ckks::YamlCKKSSchemeParam& params) {
    io.mapRequired("logN", params.logN);
    io.mapRequired("Q", params.Q);
    io.mapRequired("P", params.P);
    io.mapRequired("logDefaultScale", params.logDefaultScale);
    io.mapOptional("encryptionType", params.encryptionType,
                   mlir::heir::ckks::CKKSEncryptionType::pk);
    io.mapOptional("encryptionTechnique", params.encryptionTechnique,
                   mlir::heir::ckks::CKKSEncryptionTechnique::standard);
    io.mapOptional("moduliPerLevel", params.moduliPerLevel);
  }
};

}  // namespace yaml
}  // namespace llvm

namespace mlir {
namespace heir {
namespace ckks {

FailureOr<SchemeParamAttr> parseSchemeParamAttrFromYaml(
    MLIRContext* context, llvm::StringRef yamlStr,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  std::string diagMsg;
  auto diagHandler = [](const llvm::SMDiagnostic& diag, void* context) {
    auto* msg = static_cast<std::string*>(context);
    if (!msg->empty()) *msg += "\n";
    *msg += diag.getMessage().str();
  };

  llvm::yaml::Input yin(yamlStr, /*Ctxt=*/nullptr, diagHandler, &diagMsg);
  YamlCKKSSchemeParam params;
  yin >> params;
  if (yin.error()) {
    emitError() << "failed to parse CKKS parameter YAML: "
                << (diagMsg.empty() ? yin.error().message() : diagMsg);
    return failure();
  }

  if (params.logN <= 0) {
    emitError() << "logN must be positive, got " << params.logN;
    return failure();
  }

  DenseI64ArrayAttr qAttr = DenseI64ArrayAttr::get(context, params.Q);
  DenseI64ArrayAttr pAttr = DenseI64ArrayAttr::get(context, params.P);
  DenseI32ArrayAttr moduliPerLevelAttr =
      params.moduliPerLevel.empty()
          ? DenseI32ArrayAttr()
          : DenseI32ArrayAttr::get(context, params.moduliPerLevel);

  auto attr = SchemeParamAttr::getChecked(
      emitError, context, params.logN, qAttr, pAttr, params.logDefaultScale,
      params.encryptionType, params.encryptionTechnique, moduliPerLevelAttr);
  if (!attr) {
    return failure();
  }
  return attr;
}

FailureOr<SchemeParamAttr> loadSchemeParamAttrFromFile(
    MLIRContext* context, llvm::StringRef filename,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto buffer = llvm::MemoryBuffer::getFile(filename);
  if (!buffer) {
    emitError() << "cannot read CKKS parameter file " << filename << ": "
                << buffer.getError().message();
    return failure();
  }
  return parseSchemeParamAttrFromYaml(context, (*buffer)->getBuffer(),
                                      emitError);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
