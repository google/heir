#include "lib/Source/AutoHog/AutoHogImporter.h"

#include "include/rapidjson/document.h"      // from @rapidjson
#include "include/rapidjson/stringbuffer.h"  // from @rapidjson
#include "include/rapidjson/writer.h"        // from @rapidjson
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
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

  // Parse "ports" field to assemble input and output types
  // The function will always be a tensor<Nxi1> -> tensor<Mxi1>
  // for N = #inputs and M = #outputs
  // The maps created here will be used to access the input and output tensors
  // during operation creation.
  llvm::DenseMap<const char *, int> inputPortToTensorIndex;
  llvm::DenseMap<const char *, int> outputPortToTensorIndex;

  int numInputs = 0;
  int numOutputs = 0;

  auto &ports = document["ports"];
  assert(ports.IsObject() && "Expected 'ports' to be an object");
  for (rapidjson::Value::ConstMemberIterator itr = ports.MemberBegin();
       itr != ports.MemberEnd(); ++itr) {
    const char *portName = itr->name.GetString();
    const rapidjson::Value &port = itr->value;
    assert(port.IsObject() && "Expected port to be an object");
    const char *portDirection = port["direction"].GetString();

    if (strcmp(portDirection, "input") == 0) {
      inputPortToTensorIndex[portName] = numInputs++;
    } else if (strcmp(portDirection, "output") == 0) {
      outputPortToTensorIndex[portName] = numOutputs++;
    } else {
      llvm_unreachable("Unknown port direction, expected 'input or 'output'");
    }
  }

  Type inputType = RankedTensorType::get({numInputs}, builder.getI1Type());
  Type outputType = RankedTensorType::get({numOutputs}, builder.getI1Type());
  auto functionType = builder.getFunctionType({inputType}, {outputType});

  // FIXME: convert circuit name to legal MLIR identifier
  std::string funcName = "main";
  auto funcOp =
      builder.create<func::FuncOp>(moduleOp->getLoc(), funcName, functionType);

  builder.setInsertionPointToStart(&funcOp.getBody().front());
  BlockArgument funcArg = funcOp.getArgument(0);

  // Parse "cells" field to construct ops
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

    SmallVector<Value> operands;
    SmallVector<Type> resultTypes;

    // "port_directions" and "connections" fields are the same for all cell
    // types. "port_directions" identifies cell-local wires and describes each
    // as an input or output. "connections" maps cell-local wires to ports on
    // the module.
    DenseMap<const char *, bool> isInput;  // true if input, false if output
    DenseMap<const char *, std::string> connections;

    for (rapidjson::Value::ConstMemberIterator itr =
             cell["connections"].MemberBegin();
         itr != cell["connections"].MemberEnd(); ++itr) {
      const char *wireName = itr->name.GetString();
      // the value can be an integer or string
      std::string portName = (itr->value.IsInt())
                                 ? std::to_string(itr->value.GetInt())
                                 : std::string(itr->value.GetString());
      connections[wireName] = std::string(portName);
    }

    for (rapidjson::Value::ConstMemberIterator itr =
             cell["port_directions"].MemberBegin();
         itr != cell["port_directions"].MemberEnd(); ++itr) {
      const char *wireName = itr->name.GetString();
      const char *wireDirection = itr->value.GetString();
      isInput[wireName] = strcmp(wireDirection, "input") == 0;

      if (isInput[wireName]) {
        if (inputPortToTensorIndex.count(connections[wireName].c_str())) {
          int tensorIndex =
              inputPortToTensorIndex[connections[wireName].c_str()];
          operands.push_back(builder.create<tensor::ExtractOp>(
              funcOp.getLoc(), funcArg, builder.getIndexAttr(tensorIndex)));
        }

        // It could also be a previous cell
        // FIXME: Which cell output is not specified
      }
    }

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
