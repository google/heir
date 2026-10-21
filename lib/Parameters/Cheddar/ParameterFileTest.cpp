#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Parameters/Cheddar/ParameterFile.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FileSystem.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Path.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "tools/cpp/runfiles/runfiles.h"  // from @bazel_tools
// clang-format on

namespace mlir {
namespace heir {
namespace cheddar {
namespace {

using bazel::tools::cpp::runfiles::Runfiles;

static constexpr std::string_view kCheddarWorkspaceDir =
    "+cheddar_deps+cheddar";

class ParameterFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context.loadDialect<CheddarDialect, ckks::CKKSDialect>();
  }

  MLIRContext context;
  std::string errorMessage;

  auto getDiagHandler() {
    return [this]() -> InFlightDiagnostic {
      return emitError(UnknownLoc::get(&context));
    };
  }
};

TEST_F(ParameterFileTest, Parse32BitBootstrappingWithTerminalPrimes) {
  llvm::StringRef json = R"json({
    "log_degree": 16,
    "log_default_scale": 35,
    "boot": true,
    "dense_hamming_weight": 32768,
    "sparse_hamming_weight": 32,
    "num_cts_levels": 4,
    "num_stc_levels": 3,
    "terminal_primes": [
      30539777, 37224449
    ],
    "main_primes": [
      1060765697, 1209139201, 967180289, 1083703297
    ],
    "auxiliary_primes": [
      2147352577, 2146959361
    ],
    "default_encryption_level": 1,
    "level_config": [
      [0, 2], [2, 1], [4, 0]
    ],
    "additional_base": [0, 0]
  })json";

  auto result = parseParameterSetFromYaml(&context, json, getDiagHandler());
  ASSERT_TRUE(succeeded(result));
  ParameterSetAttr paramSet = *result;

  EXPECT_EQ(paramSet.getLogN(), 16);
  EXPECT_EQ(paramSet.getLogScale(), 35);
  EXPECT_EQ(paramSet.getWordBits(), std::optional<int>(32));
  EXPECT_EQ(paramSet.getDefaultEncryptionLevel(), std::optional<int>(1));
  EXPECT_EQ(paramSet.getDenseHammingWeight(), std::optional<int>(32768));
  EXPECT_EQ(paramSet.getSparseHammingWeight(), std::optional<int>(32));
  EXPECT_EQ(paramSet.getNumCtsLevels(), std::optional<int>(4));
  EXPECT_EQ(paramSet.getNumStcLevels(), std::optional<int>(3));

  ASSERT_TRUE(paramSet.getTerminalPrimes());
  EXPECT_EQ(paramSet.getTerminalPrimes().asArrayRef(),
            llvm::ArrayRef<int64_t>({30539777, 37224449}));
  EXPECT_EQ(paramSet.getMainPrimes().size(), 4);
  EXPECT_EQ(paramSet.getAuxPrimes().size(), 2);
  EXPECT_FALSE(paramSet.getAdditionalBase());

  // Converting to CKKS scheme_param fails because numTerminal != 0 in
  // level_config.
  auto ckksResult =
      getSchemeParamAttrFromParameterSet(paramSet, getDiagHandler());
  EXPECT_TRUE(failed(ckksResult));
}

TEST_F(ParameterFileTest, Parse32BitDoubleRescalingAndPopulateSchemeParam) {
  llvm::StringRef json = R"json({
    "log_degree": 14,
    "log_default_scale": 30,
    "default_encryption_level": 3,
    "main_primes": [
      1000000007, 1000000009, 1000000021, 1000000033, 1000000087, 1000000093
    ],
    "auxiliary_primes": [
      2000000011, 2000000033
    ],
    "level_config": [
      [2, 0], [4, 0], [6, 0]
    ]
  })json";

  auto result = parseParameterSetFromYaml(&context, json, getDiagHandler());
  ASSERT_TRUE(succeeded(result));
  ParameterSetAttr paramSet = *result;

  EXPECT_EQ(paramSet.getLogN(), 14);
  EXPECT_EQ(paramSet.getLogScale(), 30);
  EXPECT_EQ(paramSet.getWordBits(), std::optional<int>(32));
  EXPECT_EQ(paramSet.getDefaultEncryptionLevel(), std::optional<int>(3));
  EXPECT_FALSE(paramSet.getTerminalPrimes());

  auto schemeParamResult =
      getSchemeParamAttrFromParameterSet(paramSet, getDiagHandler());
  ASSERT_TRUE(succeeded(schemeParamResult));
  ckks::SchemeParamAttr schemeParam = *schemeParamResult;

  EXPECT_EQ(schemeParam.getLogN(), 14);
  EXPECT_EQ(schemeParam.getLogDefaultScale(), 30);
  EXPECT_EQ(schemeParam.getEncryptionType(), ckks::CKKSEncryptionType::pk);
  EXPECT_EQ(schemeParam.getEncryptionTechnique(),
            ckks::CKKSEncryptionTechnique::extended);

  ASSERT_TRUE(schemeParam.getModuliPerLevel());
  EXPECT_EQ(schemeParam.getModuliPerLevel().asArrayRef(),
            llvm::ArrayRef<int32_t>({2, 2, 2}));
  EXPECT_EQ(schemeParam.getQ().asArrayRef(),
            llvm::ArrayRef<int64_t>({1000000007, 1000000009, 1000000021,
                                     1000000033, 1000000087, 1000000093}));
  EXPECT_EQ(schemeParam.getP().asArrayRef(),
            llvm::ArrayRef<int64_t>({2000000011, 2000000033}));

  // Test populateSchemeParamFromParameterSet on ModuleOp.
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  ASSERT_TRUE(
      succeeded(populateSchemeParamFromParameterSet(*module, paramSet)));

  auto attachedParamSet = (*module)->getAttrOfType<ParameterSetAttr>(
      cheddar::kParameterSetAttrName);
  ASSERT_TRUE(attachedParamSet);
  EXPECT_EQ(attachedParamSet, paramSet);

  auto attachedSchemeParam = (*module)->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
  ASSERT_TRUE(attachedSchemeParam);
  EXPECT_EQ(attachedSchemeParam, schemeParam);
}

TEST_F(ParameterFileTest, Parse64BitAndVerifySchemeParam) {
  llvm::StringRef json = R"json({
    "log_degree": 16,
    "log_default_scale": 40,
    "boot": true,
    "num_cts_levels": 4,
    "num_stc_levels": 3,
    "main_primes": [
      1125899908022273, 1099515691009, 1099523555329
    ],
    "auxiliary_primes": [
      72057594038321153, 72057594040680449
    ],
    "default_encryption_level": 3,
    "level_config": [
      [1, 0], [2, 0], [3, 0]
    ]
  })json";

  auto result = parseParameterSetFromYaml(&context, json, getDiagHandler());
  ASSERT_TRUE(succeeded(result));
  ParameterSetAttr paramSet = *result;

  EXPECT_EQ(paramSet.getWordBits(), std::optional<int>(64));

  auto schemeParamResult =
      getSchemeParamAttrFromParameterSet(paramSet, getDiagHandler());
  ASSERT_TRUE(succeeded(schemeParamResult));
  ckks::SchemeParamAttr schemeParam = *schemeParamResult;

  ASSERT_TRUE(schemeParam.getModuliPerLevel());
  EXPECT_EQ(schemeParam.getModuliPerLevel().asArrayRef(),
            llvm::ArrayRef<int32_t>({1, 1, 1}));
  EXPECT_EQ(schemeParam.getQ().asArrayRef(),
            llvm::ArrayRef<int64_t>(
                {1125899908022273, 1099515691009, 1099523555329}));
}

TEST_F(ParameterFileTest, LoadParameterSetFromFile) {
  llvm::SmallString<128> tempPath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("cheddar_param", "json", tempPath));

  std::error_code ec;
  llvm::raw_fd_ostream os(tempPath, ec);
  ASSERT_FALSE(ec);

  os << R"json({
    "log_degree": 12,
    "log_default_scale": 30,
    "default_encryption_level": 1,
    "main_primes": [1000000007],
    "auxiliary_primes": [2000000011],
    "level_config": [[1, 0]]
  })json";
  os.close();

  auto result = loadParameterSetFromFile(&context, tempPath, getDiagHandler());
  llvm::sys::fs::remove(tempPath);

  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->getLogN(), 12);
  EXPECT_EQ(result->getLogScale(), 30);
}

TEST_F(ParameterFileTest, LoadUpstreamCheddarParameterFiles) {
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest());
  ASSERT_NE(runfiles, nullptr);

  auto getRunfilePath = [&](llvm::StringRef relPath) {
    llvm::SmallString<128> workspaceRelativePath;
    llvm::sys::path::append(workspaceRelativePath, kCheddarWorkspaceDir,
                            relPath);
    return runfiles->Rlocation(std::string(workspaceRelativePath.str()));
  };

  // 1. bootparam_30.json: 32-bit primes, no terminal_primes, 35 levels.
  {
    auto paramSet = loadParameterSetFromFile(
        &context, getRunfilePath("parameters/bootparam_30.json"),
        getDiagHandler());
    ASSERT_TRUE(succeeded(paramSet));
    EXPECT_EQ(paramSet->getLogN(), 16);
    EXPECT_EQ(paramSet->getLogScale(), 30);
    EXPECT_EQ(paramSet->getWordBits(), std::optional<int>(32));
    EXPECT_EQ(paramSet->getDefaultEncryptionLevel(), std::optional<int>(22));
    EXPECT_EQ(paramSet->getMainPrimes().size(), 48);
    EXPECT_EQ(paramSet->getAuxPrimes().size(), 12);
    EXPECT_FALSE(paramSet->getTerminalPrimes());

    auto schemeParam =
        getSchemeParamAttrFromParameterSet(*paramSet, getDiagHandler());
    ASSERT_TRUE(succeeded(schemeParam));
    EXPECT_EQ(schemeParam->getQ().size(), 48);
    EXPECT_EQ(schemeParam->getP().size(), 12);
    ASSERT_TRUE(schemeParam->getModuliPerLevel());
    EXPECT_EQ(schemeParam->getModuliPerLevel().size(), 35);
  }

  // 2. bootparam_35.json & bootparam_40.json: 32-bit primes with
  // terminal_primes.
  for (llvm::StringRef file :
       {"parameters/bootparam_35.json", "parameters/bootparam_40.json"}) {
    auto paramSet = loadParameterSetFromFile(&context, getRunfilePath(file),
                                             getDiagHandler());
    ASSERT_TRUE(succeeded(paramSet));
    EXPECT_EQ(paramSet->getLogN(), 16);
    EXPECT_EQ(paramSet->getWordBits(), std::optional<int>(32));
    EXPECT_TRUE(paramSet->getTerminalPrimes());
  }

  // 3. bootparam_40_64bit.json: 64-bit primes, no terminal_primes, 26 levels.
  {
    auto paramSet = loadParameterSetFromFile(
        &context, getRunfilePath("parameters/bootparam_40_64bit.json"),
        getDiagHandler());
    ASSERT_TRUE(succeeded(paramSet));
    EXPECT_EQ(paramSet->getLogN(), 16);
    EXPECT_EQ(paramSet->getLogScale(), 40);
    EXPECT_EQ(paramSet->getWordBits(), std::optional<int>(64));
    EXPECT_EQ(paramSet->getDefaultEncryptionLevel(), std::optional<int>(13));
    EXPECT_EQ(paramSet->getMainPrimes().size(), 26);
    EXPECT_EQ(paramSet->getAuxPrimes().size(), 7);
    EXPECT_FALSE(paramSet->getTerminalPrimes());

    auto schemeParam =
        getSchemeParamAttrFromParameterSet(*paramSet, getDiagHandler());
    ASSERT_TRUE(succeeded(schemeParam));
    EXPECT_EQ(schemeParam->getQ().size(), 26);
    EXPECT_EQ(schemeParam->getP().size(), 7);
    ASSERT_TRUE(schemeParam->getModuliPerLevel());
    EXPECT_EQ(schemeParam->getModuliPerLevel().size(), 26);
  }
}

TEST_F(ParameterFileTest, ErrorCases) {
  // Malformed JSON.
  EXPECT_TRUE(failed(parseParameterSetFromYaml(
      &context, "{ not valid json: ", getDiagHandler())));

  // Missing required field (log_degree).
  llvm::StringRef missingField = R"json({
    "log_default_scale": 30,
    "default_encryption_level": 1,
    "main_primes": [1000],
    "auxiliary_primes": [2000],
    "level_config": [[1, 0]]
  })json";
  EXPECT_TRUE(failed(
      parseParameterSetFromYaml(&context, missingField, getDiagHandler())));

  // Boot true but missing num_cts_levels / num_stc_levels.
  llvm::StringRef missingBootLevels = R"json({
    "log_degree": 14,
    "log_default_scale": 30,
    "default_encryption_level": 1,
    "boot": true,
    "main_primes": [1000],
    "auxiliary_primes": [2000],
    "level_config": [[1, 0]]
  })json";
  EXPECT_TRUE(failed(parseParameterSetFromYaml(&context, missingBootLevels,
                                               getDiagHandler())));

  // Invalid level_config entry size (3 instead of 2).
  llvm::StringRef invalidLevelConfigSize = R"json({
    "log_degree": 14,
    "log_default_scale": 30,
    "default_encryption_level": 1,
    "main_primes": [1000],
    "auxiliary_primes": [2000],
    "level_config": [[1, 0, 0]]
  })json";
  EXPECT_TRUE(failed(parseParameterSetFromYaml(&context, invalidLevelConfigSize,
                                               getDiagHandler())));

  // Non-strictly increasing main prime count in level_config when populating
  // ckks.scheme_param.
  llvm::StringRef nonIncreasingMain = R"json({
    "log_degree": 14,
    "log_default_scale": 30,
    "default_encryption_level": 1,
    "main_primes": [1000, 2000],
    "auxiliary_primes": [3000],
    "level_config": [[2, 0], [1, 0]]
  })json";
  auto nonIncreasingParamSet =
      parseParameterSetFromYaml(&context, nonIncreasingMain, getDiagHandler());
  ASSERT_TRUE(succeeded(nonIncreasingParamSet));
  EXPECT_TRUE(failed(getSchemeParamAttrFromParameterSet(*nonIncreasingParamSet,
                                                        getDiagHandler())));
}

}  // namespace
}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
