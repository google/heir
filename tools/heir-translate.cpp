#include "include/Target/Metadata/MetadataEmitter.h"
#include "include/Target/TfheRust/TfheRustEmitter.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  // Verilog output
  mlir::heir::registerToVerilogTranslation();
  mlir::heir::registerMetadataEmitter();

  // tfhe-rs output
  mlir::heir::tfhe_rust::registerToTfheRustTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
