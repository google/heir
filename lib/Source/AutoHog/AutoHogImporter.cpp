#include "lib/Source/AutoHog/AutoHogImporter.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Graph/Graph.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/StringMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

// JSON headers separated to prevent copybara reordering.
#include "rapidjson/document.h"      // from @rapidjson
#include "rapidjson/reader.h"        // from @rapidjson
#include "rapidjson/stringbuffer.h"  // from @rapidjson
#include "rapidjson/writer.h"        // from @rapidjson

#define DEBUG_TYPE "autohog-importer"

namespace mlir {
namespace heir {

using llvm::StringMap;

void registerFromAutoHogTranslation() {
  TranslateToMLIRRegistration reg(
      "import-autohog", "Import from AutoHoG JSON to HEIR MLIR",
      [](llvm::StringRef inputString,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        return translateFromAutoHog(inputString, context);
      },
      [](DialectRegistry &registry) {
        // Upstream dialects
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        tensor::TensorDialect>();
        // HEIR dialects
        registry.insert<cggi::CGGIDialect, lwe::LWEDialect>();
      });
}

void parseIOPorts(const rapidjson::Value &document, StringMap<Port> &ports) {
  LLVM_DEBUG(llvm::dbgs() << "Parsing IO ports\n");
  auto &docPorts = document["ports"];
  assert(docPorts.IsArray() && "Expected 'ports' to be an array");
  for (rapidjson::Value::ConstValueIterator itr = docPorts.Begin();
       itr != docPorts.End(); ++itr) {
    const rapidjson::Value &port = *itr;
    const char *portName = port["name"].GetString();
    LLVM_DEBUG(llvm::dbgs() << "Parsing IO port: " << portName << "\n");
    const char *portDirection = port["direction"].GetString();

    Port ioPort;
    ioPort.portName = std::string(portName);

    if (strcmp(portDirection, "input") == 0) {
      ioPort.type = PortType::INPUT;
    } else if (strcmp(portDirection, "output") == 0) {
      ioPort.type = PortType::OUTPUT;
    } else {
      assert(false && "Unknown port direction, expected 'input or 'output'");
    }

    ports[ioPort.portName] = ioPort;
  }

  // Second iteration to check for cell/vs IO port connection.
  // Confusing because "bits" field can be referring to either an io port
  // or a cell name.
  for (rapidjson::Value::ConstValueIterator itr = docPorts.Begin();
       itr != docPorts.End(); ++itr) {
    const rapidjson::Value &port = *itr;
    const char *portName = port["name"].GetString();
    Port ioPort = ports[portName];
    LLVM_DEBUG(llvm::dbgs() << "Checking if IO port " << portName
                            << " has a direct IO connection\n");

    if (ioPort.type == PortType::OUTPUT) {
      if (port.HasMember("bits")) {
        auto bits = port["bits"].GetArray();
        assert(bits.Size() == 1 && "Expected 'bits' to be an array of size 1");
        auto *connectedPortName = bits[0].GetString();
        if (ports.contains(connectedPortName) &&
            ports[connectedPortName].type == PortType::INPUT) {
          LLVM_DEBUG(llvm::dbgs() << "IO port " << portName
                                  << " has a direct IO connection to "
                                  << connectedPortName << "\n");
          ioPort.connectedInputPort = connectedPortName;
        }
      }
    }

    ports[ioPort.portName] = ioPort;
  }
}

/// Parse the connections field of a cell object in the JSON document.
/// - Populates the connections map, mapping the local port name to the
/// connected
///   Port object (an IO port or another cell).
/// - Populates the outputPortToConnectedCellPort map, mapping an output ioPort
///   to the corresponding local CELL type Port of the current cell.
void parseCellConnections(const char *cellName,
                          const rapidjson::Value &cellData,
                          const StringMap<Port> &ioPorts,
                          StringMap<Port> &connections,
                          StringMap<Port> &outputPortToConnectedCellPort) {
  LLVM_DEBUG(llvm::dbgs() << "Parsing cell connections\n");
  auto &jsonConnections = cellData["connections"];
  assert(jsonConnections.IsObject() &&
         "Expected 'connections' to be an object");
  for (rapidjson::Value::ConstMemberIterator itr =
           jsonConnections.MemberBegin();
       itr != jsonConnections.MemberEnd(); ++itr) {
    const char *localPortName = itr->name.GetString();
    const rapidjson::Value &portData = itr->value;
    assert(portData.IsObject() &&
           "Expected 'connections[...]' to be an object");
    const char *connectedPortName = portData["port"].GetString();

    Port port;
    port.portName = connectedPortName;

    if (portData.HasMember("cell")) {
      port.type = PortType::CELL;
      const char *connectedCell = portData["cell"].GetString();
      port.cellName = connectedCell;
    } else {
      port.type = ioPorts.at(connectedPortName).type;
      if (port.type == PortType::OUTPUT) {
        Port localPort(PortType::CELL, std::string(localPortName),
                       std::string(cellName));
        outputPortToConnectedCellPort[StringRef(connectedPortName)] = localPort;
      }
    }
    connections[StringRef(localPortName)] = port;
  }
}

/// Construct the cell graph from the JSON document.
/// - Returns a graph::Graph<const char *> object where the nodes are cell names
///   and the edges are dependencies between cells.
graph::Graph<std::string> constructCellGraph(
    const rapidjson::Document &document) {
  LLVM_DEBUG(llvm::dbgs() << "Topologically sorting cells\n");
  graph::Graph<std::string> cellGraph;
  auto &cells = document["cells"];
  assert(cells.IsObject() && "Expected 'cells' to be an object");
  for (rapidjson::Value::ConstMemberIterator itr = cells.MemberBegin();
       itr != cells.MemberEnd(); ++itr) {
    const rapidjson::Value &cell = itr->value;
    assert(cell.IsObject() && "Expected cell to be an object");
    const char *cellName = cell["cell_name"].GetString();
    cellGraph.addVertex(std::string(cellName));
    auto &connections = cell["connections"];
    assert(connections.IsObject() && "Expected 'connections' to be an object");
    for (rapidjson::Value::ConstMemberIterator itr = connections.MemberBegin();
         itr != connections.MemberEnd(); ++itr) {
      const char *localPortName = itr->name.GetString();
      const rapidjson::Value &portData = itr->value;
      assert(portData.IsObject() &&
             "Expected 'connections[...]' to be an object");
      bool isInput = strcmp(cell["port_directions"][localPortName].GetString(),
                            "input") == 0;
      if (portData.HasMember("cell") && isInput) {
        const char *connectedCell = portData["cell"].GetString();
        // Since the input is not toposorted, we may need to add the incident
        // vertex.
        if (!cellGraph.contains(std::string(connectedCell))) {
          cellGraph.addVertex(std::string(connectedCell));
        }
        LLVM_DEBUG(llvm::dbgs() << "Adding edge from " << cellName << " to "
                                << connectedCell << "\n");
        cellGraph.addEdge(std::string(cellName), std::string(connectedCell));
      }
    }
  }
  return cellGraph;
}

/// Parse a list of integers into a single LUT integer, with the first entry
/// in the list being the LUT value for the input 0.
///
/// The output integer X is interpreted as a LUT via f(input) = (X >> input) & 1
///
/// E.g., [1, 0, 1, 0] -> 5
///
int32_t parseLut(const rapidjson::GenericArray<true, rapidjson::Value> &lut) {
  int32_t result = 0;
  for (size_t i = 0; i < lut.Size(); i++) {
    int32_t bit = lut[i].GetInt();
    result |= bit << i;
  }
  return result;
}

// Parse the coefficients field of a cell object in the JSON document.
// Populates the result vector with the coefficients in the order of the
// operands.
//
// Args:
// - coefficients: a map from port name to coefficient value.
// - inputPortNameToOperandIndex: a map from port name to the
//   index of the operand in the operands vector.
// - result: the pre-resized vector to populate with the coefficients.
void parseLinCombCoefficients(
    const rapidjson::GenericObject<true, rapidjson::Value> &coefficients,
    const StringMap<int> &inputPortNameToOperandIndex,
    SmallVector<int> &result) {
  for (rapidjson::Value::ConstMemberIterator itr = coefficients.MemberBegin();
       itr != coefficients.MemberEnd(); ++itr) {
    const char *portName = itr->name.GetString();
    const int coeffValue = itr->value.GetInt();
    assert(inputPortNameToOperandIndex.contains(portName) &&
           "Coefficient port name not found in inputPortNameToOperandIndex");
    result[inputPortNameToOperandIndex.at(portName)] = coeffValue;
  }
}

OwningOpRef<Operation *> translateFromAutoHog(llvm::StringRef inputString,
                                              MLIRContext *context) {
  // Have to manually load dialects because they are loaded at parse time,
  // but we have no MLIR inputs.
  context->getOrLoadDialect<arith::ArithDialect>();
  context->getOrLoadDialect<cggi::CGGIDialect>();
  context->getOrLoadDialect<func::FuncDialect>();
  context->getOrLoadDialect<lwe::LWEDialect>();
  context->getOrLoadDialect<tensor::TensorDialect>();

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
  assert(document.IsObject() && "JSON failed to parse");

  StringMap<Port> ioPorts;
  parseIOPorts(document, ioPorts);

  OpBuilder builder(context);
  ModuleOp moduleOp = builder.create<ModuleOp>(builder.getUnknownLoc());
  OwningOpRef<Operation *> opRef(moduleOp);
  moduleOp->setAttr(
      "cggi.circuit_name",
      StringAttr::get(context, document["circuit_name"].GetString()));

  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());

  // Parse "ports" field to assemble input and output types
  // The function will always be a tensor<Nxi1> -> tensor<Mxi1>
  // for N = #inputs and M = #outputs
  // The maps created here will be used to access the input and output tensors
  // during operation creation.
  StringMap<int> inputPortToTensorIndex;
  StringMap<int> outputPortToTensorIndex;
  int numInputs = 0;
  int numOutputs = 0;
  for (auto &[portName, port] : ioPorts) {
    if (port.type == PortType::INPUT) {
      inputPortToTensorIndex[portName] = numInputs++;
    } else {
      outputPortToTensorIndex[portName] = numOutputs++;
    }
  }

  // TODO(#686): detect proper minBitWidth from the circuit
  int minBitWidth = 3;
  Type ciphertextType = lwe::LWECiphertextType::get(
      context, lwe::UnspecifiedBitFieldEncodingAttr::get(context, minBitWidth),
      lwe::LWEParamsAttr());
  Type inputType = RankedTensorType::get({numInputs}, ciphertextType);
  Type outputType = RankedTensorType::get({numOutputs}, ciphertextType);
  auto functionType = builder.getFunctionType({inputType}, {outputType});

  std::string funcName = std::string(document["circuit_name"].GetString());
  auto funcOp =
      builder.create<func::FuncOp>(moduleOp->getLoc(), funcName, functionType);
  funcOp.setPrivate();
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  BlockArgument funcArg = funcOp.getArgument(0);

  // Parse "cells" field to construct ops
  StringMap<Operation *> cellsByName;
  // Keep track of which result index a cell's output port corresponds to.
  StringMap<StringMap<int>> cellOutputPortToResultIndex;
  StringMap<Port> outputPortToConnectedCellPort;

  auto &cells = document["cells"];
  assert(cells.IsObject() && "Expected 'cells' to be an object");

  // First we must topologically sort the cells to ensure that we create
  // Operations for all inputs to a cell before visiting the cell itself.
  graph::Graph<std::string> cellGraph = constructCellGraph(document);
  auto sortResult = cellGraph.topologicalSort();
  assert(succeeded(sortResult) && "Circuit contains a cycle in its cell graph");
  LLVM_DEBUG({
    llvm::dbgs() << "Topological sort of cells:\n";
    for (std::string &cellName : sortResult.value()) {
      llvm::dbgs() << "  " << cellName << "\n";
    }
  });

  for (std::string &cellName : llvm::reverse(sortResult.value())) {
    const rapidjson::Value &cell = cells[cellName.c_str()];
    assert(cell.IsObject() && "Expected cell to be an object");

    const char *cellType = cell["type"].GetString();
    Operation *op = nullptr;

    SmallVector<Value> operands;
    // Because all the JSON fields are objects, iteration order is arbitrary,
    // and we need this to ensure the coefficient order matches the operand
    // order.
    StringMap<int> inputPortNameToOperandIndex;
    SmallVector<Type> resultTypes;

    // Cell-local port mapping to input/output
    StringMap<bool> isInput;
    for (rapidjson::Value::ConstMemberIterator itr =
             cell["port_directions"].MemberBegin();
         itr != cell["port_directions"].MemberEnd(); ++itr) {
      const char *wireName = itr->name.GetString();
      const char *wireDirection = itr->value.GetString();
      isInput[StringRef(wireName)] = strcmp(wireDirection, "input") == 0;
    }

    // Collect operands and result types for the new op
    StringMap<Port> connections;
    int resultIndex = 0;
    int operandIndex = 0;
    parseCellConnections(cellName.c_str(), cell, ioPorts, connections,
                         outputPortToConnectedCellPort);

    LLVM_DEBUG(llvm::dbgs() << "Collecting operands and results for cell: "
                            << cellName << " of type " << cellType << "\n");
    for (const auto &[localPortName, connectedPort] : connections) {
      if (isInput[localPortName]) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Processing local input port: " << localPortName << "\n");
        if (connectedPort.type == PortType::INPUT) {
          int tensorIndex = inputPortToTensorIndex[connectedPort.portName];
          auto indexValue =
              builder
                  .create<arith::ConstantIndexOp>(funcOp.getLoc(), tensorIndex)
                  .getResult();
          Value operand = builder.create<tensor::ExtractOp>(
              funcOp.getLoc(), funcArg, indexValue);
          operands.push_back(operand);
          inputPortNameToOperandIndex[localPortName] = operandIndex++;
        } else if (connectedPort.type == PortType::CELL) {
          const char *connectedCellName =
              connectedPort.cellName.value().c_str();
          if (!cellsByName.contains(connectedCellName)) {
            llvm::errs() << "Cell " << cellName
                         << " refers to unknown input cell "
                         << connectedCellName
                         << ", maybe topological sort failed or input parsing "
                            "failed, try debug mode for more info.";
            return opRef;
          }
          Operation *upstreamOp = cellsByName[connectedCellName];
          Value operand =
              upstreamOp->getResult(cellOutputPortToResultIndex
                                        .at(connectedPort.cellName.value_or(""))
                                        .at(connectedPort.portName));
          operands.push_back(operand);
          inputPortNameToOperandIndex[localPortName] = operandIndex++;
        } else {
          llvm::errs() << "Detected invalid JSON input: Cell input may not be "
                          "an output port\n";
          return opRef;
        }
      } else {  // wire is a cell output
        LLVM_DEBUG(llvm::dbgs() << "Processing local output port: "
                                << localPortName << "\n");
        resultTypes.push_back(ciphertextType);
        cellOutputPortToResultIndex[cellName][localPortName] = resultIndex;
        resultIndex++;
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Operands:\n";
      for (Value operand : operands) {
        operand.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "Results:\n";
      for (Type resultType : resultTypes) {
        resultType.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    });

    // Actually construct the op
    if (strcmp(cellType, "HomGateM") == 0) {
      SmallVector<int> coefficients;
      coefficients.resize(operands.size());
      parseLinCombCoefficients(cell["weights"].GetObject(),
                               inputPortNameToOperandIndex, coefficients);
      SmallVector<int32_t> lookupTables;
      const auto &jsonTables = cell["tableT"];
      assert(jsonTables.IsObject() && "Expected 'tableT' to be an object");

      for (rapidjson::Value::ConstMemberIterator itr = jsonTables.MemberBegin();
           itr != jsonTables.MemberEnd(); ++itr) {
        [[maybe_unused]] const char *portName = itr->name.GetString();
        const rapidjson::Value &table = itr->value;
        assert(!isInput[portName] &&
               "Expected key to tableT to be an output port");
        assert(table.IsArray() && "Expected 'tableT' to be an array");
        int lookupTable = parseLut(table.GetArray());
        lookupTables.push_back(lookupTable);
      }
      op = builder.create<cggi::MultiLutLinCombOp>(
          funcOp.getLoc(), resultTypes, operands, coefficients,
          builder.getDenseI32ArrayAttr(lookupTables));
    } else if (strcmp(cellType, "HomGateS") == 0) {
      SmallVector<int> coefficients;
      coefficients.resize(operands.size());
      parseLinCombCoefficients(cell["weights"].GetObject(),
                               inputPortNameToOperandIndex, coefficients);
      // Only one output port, so just take the first LUT
      auto lookupTableJson = cell["tableT"].MemberBegin()->value.GetArray();
      int lookupTable = parseLut(lookupTableJson);
      op = builder.create<cggi::LutLinCombOp>(
          funcOp.getLoc(), resultTypes, operands, coefficients,
          builder.getI32IntegerAttr(lookupTable));
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
      llvm::errs() << "Detected invalid JSON input: unknown cell type: "
                   << cellType << "\n";
      return opRef;
    }

    cellsByName[cellName] = op;
  }

  LLVM_DEBUG(llvm::dbgs() << "Ops created for circuit cells. Func (which "
                             "should not verify because it has no return): "
                          << funcOp << "\n");

  LLVM_DEBUG({
    llvm::dbgs() << "cellOutputPortToResultIndex=\n";
    for (auto &[cellName, portToIndex] : cellOutputPortToResultIndex) {
      llvm::dbgs() << "  " << cellName << ":\n";
      for (auto &[portName, index] : portToIndex) {
        llvm::dbgs() << "    " << portName << " -> " << index << "\n";
      }
    }
  });

  SmallVector<Value> outputValues;
  outputValues.resize(numOutputs);
  for (auto &[portName, port] : ioPorts) {
    if (port.type != PortType::OUTPUT) continue;
    LLVM_DEBUG(llvm::dbgs()
               << "Finding value for output port: " << portName << "\n");

    if (port.connectedInputPort.has_value()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Output port " << portName << " is connected to input port "
                 << port.connectedInputPort.value() << "\n");
      auto indexOp = builder.create<arith::ConstantIndexOp>(
          funcOp.getLoc(),
          inputPortToTensorIndex[port.connectedInputPort.value()]);
      outputValues[outputPortToTensorIndex[portName]] =
          builder.create<tensor::ExtractOp>(funcOp.getLoc(), funcArg,
                                            indexOp.getResult());
      continue;
    }

    if (outputPortToConnectedCellPort.contains(portName)) {
      Port cellPort = outputPortToConnectedCellPort[portName];
      LLVM_DEBUG(llvm::dbgs()
                 << "Output port " << portName << " is connected to cell port "
                 << cellPort.portName << " of cell "
                 << cellPort.cellName.value_or("") << "\n");
      Operation *op = cellsByName[cellPort.cellName.value().c_str()];
      int resultIndex =
          cellOutputPortToResultIndex.at(cellPort.cellName.value_or(""))
              .at(cellPort.portName);
      outputValues[outputPortToTensorIndex[portName]] =
          op->getResult(resultIndex);
      continue;
    }

    llvm::errs() << "Detected invalid JSON input: Output port " << portName
                 << " is not connected to any input or cell output\n";
    return opRef;
  }

  auto fromElementsOp =
      builder.create<tensor::FromElementsOp>(funcOp.getLoc(), outputValues);
  builder.create<func::ReturnOp>(funcOp.getLoc(), fromElementsOp.getResult());
  return opRef;
}

}  // namespace heir
}  // namespace mlir
