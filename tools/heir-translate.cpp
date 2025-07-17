#include "lib/Source/AutoHog/AutoHogImporter.h"
#include "lib/Target/FunctionInfo/FunctionInfoEmitter.h"
#include "lib/Target/Jaxite/JaxiteEmitter.h"
#include "lib/Target/JaxiteWord/JaxiteWordEmitter.h"
#include "lib/Target/Lattigo/LattigoEmitter.h"
#include "lib/Target/Metadata/MetadataEmitter.h"
#include "lib/Target/OpenFhePke/OpenFheTranslateRegistration.h"
// This comment includes internal emitters
#include "lib/Target/SimFHE/SimFHEEmitter.h"
#include "lib/Target/TfheRust/TfheRustEmitter.h"
#include "lib/Target/TfheRustBool/TfheRustBoolEmitter.h"
#include "lib/Target/TfheRustHL/TfheRustHLEmitter.h"
#include "lib/Target/Verilog/VerilogEmitter.h"
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  // Verilog output
  mlir::heir::registerToVerilogTranslation();
  mlir::heir::registerMetadataEmitter();

  // tfhe-rs output
  mlir::heir::tfhe_rust::registerToTfheRustTranslation();
  mlir::heir::tfhe_rust::registerToTfheRustHLTranslation();
  mlir::heir::tfhe_rust_bool::registerToTfheRustBoolTranslation();

  // jaxite output
  mlir::heir::jaxite::registerToJaxiteTranslation();
  mlir::heir::jaxiteword::registerToJaxiteWordTranslation();

  // Misc
  mlir::heir::simfhe::registerToSimFHETranslation();
  mlir::heir::functioninfo::registerToFunctionInfoTranslation();

  // OpenFHE
  mlir::heir::openfhe::registerTranslateOptions();
  mlir::heir::openfhe::registerToOpenFhePkeTranslation();
  mlir::heir::openfhe::registerToOpenFhePkeHeaderTranslation();
  mlir::heir::openfhe::registerToOpenFhePkePybindTranslation();

  // Lattigo
  mlir::heir::lattigo::registerToLattigoTranslation();
  mlir::heir::lattigo::registerTranslateOptions();

  // AutoHOG input
  mlir::heir::registerFromAutoHogTranslation();

  // This comment inserts internal emitters

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
