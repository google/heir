#include "lib/Source/AutoHog/AutoHogImporter.h"
#include "lib/Target/Jaxite/JaxiteEmitter.h"
#include "lib/Target/Metadata/MetadataEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeEmitter.h"
#include "lib/Target/OpenFhePke/OpenFhePkeHeaderEmitter.h"
#include "lib/Target/TfheRust/TfheRustEmitter.h"
#include "lib/Target/TfheRustBool/TfheRustBoolEmitter.h"
#include "lib/Target/Verilog/VerilogEmitter.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  // Verilog output
  mlir::heir::registerToVerilogTranslation();
  mlir::heir::registerMetadataEmitter();

  // tfhe-rs output
  mlir::heir::tfhe_rust::registerToTfheRustTranslation();
  mlir::heir::tfhe_rust_bool::registerToTfheRustBoolTranslation();

  // jaxite output
  mlir::heir::jaxite::registerToJaxiteTranslation();

  // OpenFHE
  mlir::heir::openfhe::registerToOpenFhePkeTranslation();
  mlir::heir::openfhe::registerToOpenFhePkeHeaderTranslation();

  // AutoHOG input
  mlir::heir::registerFromAutoHogTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
