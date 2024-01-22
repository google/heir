#include "include/Target/Metadata/MetadataEmitter.h"
#include "include/Target/OpenFhePke/OpenFhePkeEmitter.h"
#include "include/Target/TfheRust/TfheRustEmitter.h"
#include "include/Target/TfheRustBool/TfheRustBoolEmitter.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  // Verilog output
  mlir::heir::registerToVerilogTranslation();
  mlir::heir::registerMetadataEmitter();

  // tfhe-rs output
  mlir::heir::tfhe_rust::registerToTfheRustTranslation();
  mlir::heir::tfhe_rust_bool::registerToTfheRustBoolTranslation();

  // OpenFHE
  mlir::heir::openfhe::registerToOpenFhePkeTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
