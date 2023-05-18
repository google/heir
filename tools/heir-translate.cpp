#include "include/Target/Verilog/VerilogEmitter.h"
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h" // from @llvm-project

int main(int argc, char** argv) {
  mlir::heir::registerToVerilogTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}