#include <cstdlib>

#include "lib/Tools/HeirOptRegistration.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"          // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project

#ifndef HEIR_NO_YOSYS
#include "lib/Tools/HeirYosysRegistration.h"
#endif

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::heir::registerAllHEIRDialects(registry);
  mlir::heir::registerAllHEIRPasses();

  // Register the Yosys optimizer pipeline, unless heir-opt is built without
  // Yosys (`--//:enable_yosys=0`, as for the python wheel).
#ifndef HEIR_NO_YOSYS
#if !defined(HEIR_ABC_BINARY) || !defined(HEIR_YOSYS_SCRIPTS_DIR)
#error "HEIR_ABC_BINARY and HEIR_YOSYS_SCRIPTS_DIR must be defined"
#endif
  const char* abcEnvPath = HEIR_ABC_BINARY;
  const char* yosysRunfilesEnvPath = HEIR_YOSYS_SCRIPTS_DIR;
  // When running in a lit test, these #defines must be overridden
  // by environment variables set in tests/lit.cfg.py
  char* overriddenAbcEnvPath = std::getenv("HEIR_ABC_BINARY");
  char* overriddenYosysRunfilesEnvPath = std::getenv("HEIR_YOSYS_SCRIPTS_DIR");
  if (overriddenAbcEnvPath != nullptr) abcEnvPath = overriddenAbcEnvPath;
  if (overriddenYosysRunfilesEnvPath != nullptr)
    yosysRunfilesEnvPath = overriddenYosysRunfilesEnvPath;
  mlir::heir::registerHEIRYosysPipelines(yosysRunfilesEnvPath, abcEnvPath);
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
