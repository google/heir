#include "lib/Target/Lattigo/LattigoTranslateRegistration.h"

#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/Lattigo/LattigoDebugEmitter.h"
#include "lib/Target/Lattigo/LattigoEmitter.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/TargetUtils.h"
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
namespace lattigo {

struct TranslateOptions {
  llvm::cl::opt<std::string> packageName{
      "package-name",
      llvm::cl::desc("The name to use for the package declaration in the "
                     "generated golang file."),
      llvm::cl::init("main")};
  llvm::cl::list<std::string> extraImports{
      "extra-imports", llvm::cl::desc("Additional import paths")};
};

static llvm::ManagedStatic<TranslateOptions> translateOptions;

void registerTranslateOptions() {
  // Forces initialization of options.
  *translateOptions;
}

void registerToLattigoTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-lattigo",
      "translate the lattigo dialect to GO code against the Lattigo API",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToLattigo(op, output, translateOptions->packageName,
                                  translateOptions->extraImports);
      },
      [](DialectRegistry& registry) {
        registry
            .insert<affine::AffineDialect, rns::RNSDialect, arith::ArithDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    tensor_ext::TensorExtDialect, lattigo::LattigoDialect,
                    mgmt::MgmtDialect, scf::SCFDialect>();
      });
}

void registerToLattigoDebugTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-lattigo-debug",
      "Emit source code containing default debug helper implementation for "
      "lattigo dialect",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToDebugEmitter(op, output,
                                       translateOptions->packageName);
      },
      [](DialectRegistry& registry) {
        registry
            .insert<affine::AffineDialect, rns::RNSDialect, arith::ArithDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    tensor_ext::TensorExtDialect, lattigo::LattigoDialect,
                    mgmt::MgmtDialect, scf::SCFDialect>();
      });
}

void registerToLattigoPreprocessingTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-lattigo-preprocessing",
      "translate the lattigo dialect to GO code against the Lattigo API",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToLattigo(
            op, output, translateOptions->packageName,
            translateOptions->extraImports, [](func::FuncOp funcOp) {
              return funcOp->hasAttr(kClientPackFuncAttrName);
            });
      },
      [](DialectRegistry& registry) {
        registry
            .insert<affine::AffineDialect, rns::RNSDialect, arith::ArithDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    tensor_ext::TensorExtDialect, lattigo::LattigoDialect,
                    mgmt::MgmtDialect, scf::SCFDialect>();
      });
}

void registerToLattigoPreprocessedTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-lattigo-preprocessed",
      "translate the lattigo dialect to GO code against the Lattigo API",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToLattigo(
            op, output, translateOptions->packageName,
            translateOptions->extraImports, [](func::FuncOp funcOp) {
              return !funcOp->hasAttr(kClientPackFuncAttrName);
            });
      },
      [](DialectRegistry& registry) {
        registry
            .insert<affine::AffineDialect, rns::RNSDialect, arith::ArithDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    tensor_ext::TensorExtDialect, lattigo::LattigoDialect,
                    mgmt::MgmtDialect, scf::SCFDialect>();
      });
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
