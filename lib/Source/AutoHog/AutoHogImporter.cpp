#include "lib/Source/AutoHog/AutoHogImporter.h"

#include "include/rapidjson/document.h"      // from @rapidjson
#include "include/rapidjson/stringbuffer.h"  // from @rapidjson
#include "include/rapidjson/writer.h"        // from @rapidjson
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "llvm/include/llvm/ADT/StringRef.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
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

#define DEBUG_TYPE "autohog-importer"

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
  LLVM_DEBUG(llvm::dbgs() << "Translating from AutoHoG JSON to MLIR\n");
  LLVM_DEBUG(llvm::dbgs() << "Input string: \n" << inputString << "\n");
  const char *json = inputString.data();
  rapidjson::StringStream ss(json);
  rapidjson::Document document;
  document.ParseStream(ss);

  LLVM_DEBUG({
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    const char *output = buffer.GetString();
    llvm::dbgs() << "Parsed JSON: " << output << "\n";
  });

  OpBuilder builder(context);
  Operation *op = builder.create<ModuleOp>(builder.getUnknownLoc());
  op->setAttr("cggi.circuit_name",
              StringAttr::get(context, document["circuit_name"].GetString()));
  OwningOpRef<Operation *> opRef(op);
  return opRef;
}

}  // namespace heir
}  // namespace mlir
