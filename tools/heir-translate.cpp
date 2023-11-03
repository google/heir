#include "include/Target/Metadata/MetadataEmitter.h"
#include "include/Target/TfheRust/TfheRustEmitter.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

using namespace mlir::heir;

int main(int argc, char **argv) {
  // Verilog output
  registerToVerilogTranslation();
  registerMetadataEmitter();

  // tfhe-rs output
  tfhe_rust::registerToTfheRustTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
