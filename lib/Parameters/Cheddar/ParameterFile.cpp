#include "lib/Parameters/Cheddar/ParameterFile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "llvm/include/llvm/Support/MemoryBuffer.h"     // from @llvm-project
#include "llvm/include/llvm/Support/SourceMgr.h"        // from @llvm-project
#include "llvm/include/llvm/Support/YAMLTraits.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace {

struct YamlCheddarParameterSet {
  int log_degree = 0;
  int log_default_scale = 0;
  int default_encryption_level = 0;
  std::vector<int64_t> main_primes;
  std::vector<int64_t> auxiliary_primes;
  std::vector<int64_t> terminal_primes;
  std::vector<std::vector<int32_t>> level_config;
  std::vector<int32_t> additional_base;
  std::optional<int> dense_hamming_weight;
  std::optional<int> sparse_hamming_weight;
  bool boot = false;
  std::optional<int> num_cts_levels;
  std::optional<int> num_stc_levels;
};

}  // namespace

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::vector<int32_t>)

template <>
struct llvm::yaml::MappingTraits<YamlCheddarParameterSet> {
  static void mapping(IO& io, YamlCheddarParameterSet& params) {
    io.mapRequired("log_degree", params.log_degree);
    io.mapRequired("log_default_scale", params.log_default_scale);
    io.mapRequired("default_encryption_level", params.default_encryption_level);
    io.mapRequired("main_primes", params.main_primes);
    io.mapRequired("auxiliary_primes", params.auxiliary_primes);
    io.mapOptional("terminal_primes", params.terminal_primes);
    io.mapRequired("level_config", params.level_config);
    io.mapOptional("additional_base", params.additional_base);
    io.mapOptional("dense_hamming_weight", params.dense_hamming_weight);
    io.mapOptional("sparse_hamming_weight", params.sparse_hamming_weight);
    io.mapOptional("boot", params.boot, false);
    io.mapOptional("num_cts_levels", params.num_cts_levels);
    io.mapOptional("num_stc_levels", params.num_stc_levels);
  }
};

namespace mlir {
namespace heir {
namespace cheddar {

FailureOr<ParameterSetAttr> parseParameterSetFromYaml(
    MLIRContext* context, llvm::StringRef yamlStr,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  std::string diagMsg;
  auto diagHandler = [](const llvm::SMDiagnostic& diag, void* context) {
    auto* msg = static_cast<std::string*>(context);
    if (!msg->empty()) *msg += '\n';
    *msg += diag.getMessage().str();
  };

  llvm::yaml::Input yin(yamlStr, /*Ctxt=*/nullptr, diagHandler, &diagMsg);
  YamlCheddarParameterSet params;
  yin >> params;
  if (yin.error()) {
    emitError() << "failed to parse Cheddar parameter YAML: "
                << (diagMsg.empty() ? yin.error().message() : diagMsg);
    return failure();
  }

  SmallVector<int32_t> flatLevelConfig;
  flatLevelConfig.reserve(params.level_config.size() * 2);
  for (size_t i = 0; i < params.level_config.size(); ++i) {
    if (params.level_config[i].size() != 2) {
      emitError() << "level_config entry at index " << i
                  << " must have size 2, got " << params.level_config[i].size();
      return failure();
    }
    flatLevelConfig.push_back(params.level_config[i][0]);
    flatLevelConfig.push_back(params.level_config[i][1]);
  }
  DenseI32ArrayAttr levelConfigAttr =
      DenseI32ArrayAttr::get(context, flatLevelConfig);

  int64_t maxPrime = 0;
  for (int64_t p : params.main_primes) {
    maxPrime = std::max(maxPrime, p);
  }
  for (int64_t p : params.auxiliary_primes) {
    maxPrime = std::max(maxPrime, p);
  }
  for (int64_t p : params.terminal_primes) {
    maxPrime = std::max(maxPrime, p);
  }
  int wordBits = (maxPrime < (1LL << 31)) ? 32 : 64;

  DenseI64ArrayAttr mainPrimesAttr =
      DenseI64ArrayAttr::get(context, params.main_primes);
  DenseI64ArrayAttr auxPrimesAttr =
      DenseI64ArrayAttr::get(context, params.auxiliary_primes);
  DenseI64ArrayAttr terminalPrimesAttr =
      params.terminal_primes.empty()
          ? DenseI64ArrayAttr()
          : DenseI64ArrayAttr::get(context, params.terminal_primes);

  DenseI32ArrayAttr additionalBaseAttr;
  if (!params.additional_base.empty()) {
    if (params.additional_base.size() != 2) {
      emitError() << "additional_base must have size 2, got "
                  << params.additional_base.size();
      return failure();
    }
    if (!(params.additional_base[0] == 0 && params.additional_base[1] == 0)) {
      additionalBaseAttr =
          DenseI32ArrayAttr::get(context, params.additional_base);
    }
  }

  auto attr = ParameterSetAttr::getChecked(
      emitError, context, params.log_degree, params.log_default_scale,
      mainPrimesAttr, auxPrimesAttr, terminalPrimesAttr, levelConfigAttr,
      /*wordBits=*/std::optional<int>(wordBits),
      /*defaultEncryptionLevel=*/
      std::optional<int>(params.default_encryption_level), additionalBaseAttr,
      params.dense_hamming_weight, params.sparse_hamming_weight, params.boot,
      params.num_cts_levels, params.num_stc_levels);
  if (!attr) {
    return failure();
  }
  return attr;
}

FailureOr<ParameterSetAttr> loadParameterSetFromFile(
    MLIRContext* context, llvm::StringRef filename,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto buffer = llvm::MemoryBuffer::getFile(filename);
  if (!buffer) {
    emitError() << "cannot read Cheddar parameter file " << filename << ": "
                << buffer.getError().message();
    return failure();
  }
  return parseParameterSetFromYaml(context, (*buffer)->getBuffer(), emitError);
}

FailureOr<ckks::SchemeParamAttr> getSchemeParamAttrFromParameterSet(
    ParameterSetAttr parameterSet,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext* context = parameterSet.getContext();
  DenseI64ArrayAttr mainPrimesAttr = parameterSet.getMainPrimes();
  ArrayRef<int64_t> mainPrimes = mainPrimesAttr.asArrayRef();

  DenseI64ArrayAttr qAttr;
  DenseI32ArrayAttr moduliPerLevelAttr;

  if (DenseI32ArrayAttr levelConfigAttr = parameterSet.getLevelConfig()) {
    ArrayRef<int32_t> levelConfig = levelConfigAttr.asArrayRef();
    SmallVector<int32_t> moduliPerLevel;
    int32_t prevNumMain = 0;
    for (size_t i = 0; i < levelConfig.size(); i += 2) {
      int32_t numMain = levelConfig[i];
      int32_t numTerminal = levelConfig[i + 1];

      // TODO: Support terminal_primes in level_config when populating
      // ckks.scheme_param.
      if (numTerminal != 0) {
        emitError() << "terminal_primes in level_config are not yet supported "
                       "when populating ckks.scheme_param";
        return failure();
      }

      int32_t count = numMain - prevNumMain;
      if (count <= 0) {
        emitError() << "main prime count in level_config must strictly "
                       "increase at each level";
        return failure();
      }
      moduliPerLevel.push_back(count);
      prevNumMain = numMain;
    }
    qAttr = DenseI64ArrayAttr::get(context, mainPrimes.take_front(prevNumMain));
    moduliPerLevelAttr = DenseI32ArrayAttr::get(context, moduliPerLevel);
  } else {
    qAttr = mainPrimesAttr;
    moduliPerLevelAttr = DenseI32ArrayAttr();
  }

  auto schemeParam = ckks::SchemeParamAttr::getChecked(
      emitError, context, parameterSet.getLogN(), qAttr,
      parameterSet.getAuxPrimes(), parameterSet.getLogScale(),
      ckks::CKKSEncryptionType::pk, ckks::CKKSEncryptionTechnique::extended,
      moduliPerLevelAttr);
  if (!schemeParam) {
    return failure();
  }
  return schemeParam;
}

LogicalResult populateSchemeParamFromParameterSet(
    ModuleOp module, ParameterSetAttr parameterSet) {
  auto schemeParamAttr = getSchemeParamAttrFromParameterSet(
      parameterSet, [&]() { return module->emitError(); });
  if (failed(schemeParamAttr)) {
    return failure();
  }
  module->setAttr(cheddar::kParameterSetAttrName, parameterSet);
  module->setAttr(ckks::CKKSDialect::kSchemeParamAttrName, *schemeParamAttr);
  return success();
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
