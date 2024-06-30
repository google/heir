#include "lib/Source/AutoHog/AutoHogImporter.h"

#include "include/rapidjson/document.h"      // from @rapidjson
#include "include/rapidjson/stringbuffer.h"  // from @rapidjson
#include "include/rapidjson/writer.h"        // from @rapidjson
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
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
  Operation *moduleOp = builder.create<ModuleOp>(builder.getUnknownLoc());
  moduleOp->setAttr(
      "cggi.circuit_name",
      StringAttr::get(context, document["circuit_name"].GetString()));

  builder.setInsertionPointToStart(moduleOp->getBlock());
  SmallVector<Type, 4> inputTypes;
  SmallVector<Type, 4> resultTypes;
  // FIXME; set input types and result types
  auto type = builder.getFunctionType(inputTypes, resultTypes);
  // FIXME: convert circuit name to legal MLIR identifier
  std::string funcName = "main";
  auto funcOp =
      builder.create<func::FuncOp>(moduleOp->getLoc(), funcName, type);

  builder.setInsertionPointToStart(&funcOp.getBody().front());

  llvm::DenseMap<const char *, Operation *> cellsByName;

  auto &cells = document["cells"];
  assert(cells.IsObject() && "Expected 'cells' to be an object");
  for (rapidjson::Value::ConstMemberIterator itr = cells.MemberBegin();
       itr != cells.MemberEnd(); ++itr) {
    const char *cellName = itr->name.GetString();
    const rapidjson::Value &cell = itr->value;
    assert(cell.IsObject() && "Expected cell to be an object");
    const char *cellType = cell["type"].GetString();
    Operation *op = nullptr;

    ValueRange operands;
    TypeRange resultTypes;
    // FIXME: populate operands and result types

    if (strcmp(cellType, "HomGateM") == 0) {
      llvm_unreachable("Unimplemented");
    } else if (strcmp(cellType, "HomGateS") == 0) {
      op = builder.create<cggi::LutLinCombOp>(funcOp.getLoc(), resultTypes,
                                              operands);
    } else if (strcmp(cellType, "AND") == 0) {
      op = builder.create<cggi::AndOp>(funcOp.getLoc(), resultTypes, operands);
    } else if (strcmp(cellType, "NAND") == 0) {
      op = builder.create<cggi::NandOp>(funcOp.getLoc(), resultTypes, operands);
    } else if (strcmp(cellType, "NOR") == 0) {
      op = builder.create<cggi::NorOp>(funcOp.getLoc(), resultTypes, operands);
    } else if (strcmp(cellType, "OR") == 0) {
      op = builder.create<cggi::OrOp>(funcOp.getLoc(), resultTypes, operands);
    } else if (strcmp(cellType, "XOR") == 0) {
      op = builder.create<cggi::XorOp>(funcOp.getLoc(), resultTypes, operands);
    } else if (strcmp(cellType, "XNOR") == 0) {
      op = builder.create<cggi::XNorOp>(funcOp.getLoc(), resultTypes, operands);
    } else if (strcmp(cellType, "NOT") == 0) {
      op = builder.create<cggi::NotOp>(funcOp.getLoc(), resultTypes, operands);
    } else {
      llvm_unreachable("Unknown cell type");
    }

    cellsByName[cellName] = op;
  }

  OwningOpRef<Operation *> opRef(moduleOp);
  return opRef;
}

}  // namespace heir
}  // namespace mlir
