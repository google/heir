#include "lib/Target/OpenFheEmitC/OpenFheEmitCTranslateRegistration.h"

#include "lib/Dialect/Openfhe/Conversions/OpenFHEToEmitC/OpenfheToEmitCDialectInterface.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/OpenFheEmitC/OpenFheEmitCEmitter.h"
#include "lib/Target/OpenFheEmitC/OpenFhePybindEmitter.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ArithToEmitC/ArithToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/FuncToEmitC/FuncToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/SCFToEmitC/SCFToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir::heir::openfhe {

void registerToOpenFheEmitCTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-emitc",
      "Translate OpenFHE/EmitC ops to C++ code using EmitC tools",
      [](Operation* op, llvm::raw_ostream& os) {
        return translateToOpenFheEmitC(op, os);
      },
      [](DialectRegistry& registry) {
        registry.insert<mlir::heir::openfhe::OpenfheDialect,
                        mlir::emitc::EmitCDialect, mlir::func::FuncDialect,
                        mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect,
                        mlir::heir::tensor_ext::TensorExtDialect,
                        mlir::bufferization::BufferizationDialect>();
        mlir::registerConvertArithToEmitCInterface(registry);
        mlir::registerConvertFuncToEmitCInterface(registry);
        mlir::registerConvertMemRefToEmitCInterface(registry);
        mlir::registerConvertSCFToEmitCInterface(registry);
        mlir::heir::openfhe::registerOpenfheToEmitCInterface(registry);
      });

  TranslateFromMLIRRegistration regPybind(
      "emit-openfhe-pybind",
      "Translate OpenFHE/EmitC nested pybind module to pybind11 C++ code",
      [](Operation* op, llvm::raw_ostream& os) {
        return translateToOpenFhePybind(op, os);
      },
      [](DialectRegistry& registry) {
        registry.insert<mlir::heir::openfhe::OpenfheDialect,
                        mlir::emitc::EmitCDialect, mlir::func::FuncDialect,
                        mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect,
                        mlir::heir::tensor_ext::TensorExtDialect,
                        mlir::bufferization::BufferizationDialect>();
        mlir::registerConvertArithToEmitCInterface(registry);
        mlir::registerConvertFuncToEmitCInterface(registry);
        mlir::registerConvertMemRefToEmitCInterface(registry);
        mlir::registerConvertSCFToEmitCInterface(registry);
        mlir::heir::openfhe::registerOpenfheToEmitCInterface(registry);
      });
}

}  // namespace mlir::heir::openfhe
