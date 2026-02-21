#include "lib/Target/OpenFhePke/OpenFheTranslateRegistration.h"

#include <string>

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/OpenFhePke/OpenFhePkeEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeHeaderEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeDebugHeaderEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkePybindEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeDebugEmitter.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "llvm/include/llvm/Support/CommandLine.h"    // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

struct TranslateOptions {
  llvm::cl::opt<mlir::heir::openfhe::OpenfheImportType> openfheImportType{
      "openfhe-include-type",
      llvm::cl::desc("The type of imports to use for OpenFHE"),
      llvm::cl::init(mlir::heir::openfhe::OpenfheImportType::INSTALL_RELATIVE),
      llvm::cl::values(
          clEnumValN(mlir::heir::openfhe::OpenfheImportType::INSTALL_RELATIVE,
                     "install-relative",
                     "Emit OpenFHE with install-relative import paths (default "
                     "for user-facing code)"),
          clEnumValN(mlir::heir::openfhe::OpenfheImportType::SOURCE_RELATIVE,
                     "source-relative",
                     "Emit OpenFHE with source-relative import paths (default "
                     "for HEIR-internal development)"),
          clEnumValN(mlir::heir::openfhe::OpenfheImportType::EMBEDDED,
                     "embedded",
                     "Emit OpenFHE with embedded import paths (default "
                     "for code to be included in OpenFHE source files)"))};
  llvm::cl::opt<std::string> weightsFile{
      "weights-file",
      llvm::cl::desc("Emit all dense elements attributes to this binary file")};
  llvm::cl::opt<bool> skipVectorResizing{
      "skip-vector-resizing",
      llvm::cl::desc("Skip resizing vectors to ringdimension/2 when emitting "
                     "OpenFHE PKE code, i.e., assume the dimensions in the "
                     "input IR are correct already."),
      llvm::cl::init(false)};
  llvm::cl::opt<std::string> openfheDebugHelperIncludePath{
      "openfhe-debug-helper-include-path",
      llvm::cl::desc("The path to the header defining debug helper functions")};
};
static llvm::ManagedStatic<TranslateOptions> options;

struct PybindOptions {
  llvm::cl::opt<std::string> pybindHeaderInclude{
      "pybind-header-include",
      llvm::cl::desc(
          "The HEIR-generated header to include for the pybind11 bindings")};
  llvm::cl::opt<std::string> pybindModuleName{
      "pybind-module-name",
      llvm::cl::desc(
          "The name of the generated python module (must match the .so file)")};
};
static llvm::ManagedStatic<PybindOptions> pybindOptions;

void registerTranslateOptions() {
  // Forces initialization of options.
  *options;
  *pybindOptions;
}

// Common func to register dialects
static void registerRelevantDialects(DialectRegistry& registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect,
                  openfhe::OpenfheDialect, lwe::LWEDialect,
                  tensor_ext::TensorExtDialect, polynomial::PolynomialDialect,
                  tensor::TensorDialect, mod_arith::ModArithDialect,
                  rns::RNSDialect, affine::AffineDialect, scf::SCFDialect>();
  rns::registerExternalRNSTypeInterfaces(registry);
}

void registerToOpenFhePkeTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke",
      "translate the openfhe dialect to C++ code against the OpenFHE pke API",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToOpenFhePke(op, output, options->openfheImportType,
                                     options->weightsFile,
                                     options->skipVectorResizing);
      },
      registerRelevantDialects);
}

void registerToOpenFhePkeHeaderTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke-header",
      "Emit a header corresponding to the C++ file generated by "
      "--emit-openfhe-pke",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToOpenFhePkeHeader(
            op, output, options->openfheImportType,
            options->openfheDebugHelperIncludePath);
      },
      registerRelevantDialects);
}

void registerToOpenFhePkePybindTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke-pybind",
      "Emit a C++ file containing pybind11 bindings for the input openfhe "
      "dialect IR"
      "--emit-openfhe-pke-pybind",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToOpenFhePkePybind(op, output,
                                           pybindOptions->pybindHeaderInclude,
                                           pybindOptions->pybindModuleName);
      },
      registerRelevantDialects);
}

 
void registerToOpenFhePkeDebugTranslation(){
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke-debug",
      "Emit a C++ file containing default debug helper implementation.",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToOpenFhePkeDebugEmitter(
          op, output, options->openfheImportType,
          options->openfheDebugHelperIncludePath);
      },
      registerRelevantDialects);
}


void registerToOpenFhePkeDebugHeaderTranslation(){
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke-debug-header",
      "Emit a header corresponding to the C++ file generated by " 
      "--emit-openfhe-pke-debug",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToOpenFhePkeDebugHeaderEmitter(
          op, output, options->openfheImportType);
      },
      registerRelevantDialects);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
