#include "lib/Tools/HeirTranslateRegistration.h"
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char** argv) {
  mlir::heir::registerAllHEIRTranslations();
  return failed(mlir::mlirTranslateMain(argc, argv, "HEIR Translation Tool"));
}
