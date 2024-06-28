#include "lib/Source/AutoHog/AutoHogImporter.h"

#include "include/rapidjson/document.h"  // from @rapidjson
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "llvm/include/llvm/ADT/StringRef.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"      // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {

void registerFromAutoHogTranslation() {
  TranslateToMLIRRegistration reg(
      "import-autohog", "Import from AutoHoG JSON to HEIR MLIR",
      [](llvm::StringRef inputString,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        return translateFromAutoHog(inputString, context);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, cggi::CGGIDialect>();
      });
}

OwningOpRef<Operation *> translateFromAutoHog(llvm::StringRef inputString,
                                              MLIRContext *context) {
  rapidjson::StringStream ss(inputString.str().c_str());
  rapidjson::Document document;
  document.ParseStream(ss);

  OpBuilder builder(context);
  Operation *op = builder.create<ModuleOp>(builder.getUnknownLoc());
  OwningOpRef<Operation *> opRef(op);
  return opRef;
}

}  // namespace heir
}  // namespace mlir
