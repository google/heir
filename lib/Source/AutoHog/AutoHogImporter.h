#ifndef LIB_SOURCE_AUTOHOG_AUTOHOGIMPORTER_H_
#define LIB_SOURCE_AUTOHOG_AUTOHOGIMPORTER_H_

#include <optional>
#include <string>
#include <utility>

#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {

enum class PortType { INPUT, OUTPUT, CELL };

// A port that can either be an IO port or a cell port.
struct Port {
  PortType type;
  std::string portName;
  std::optional<std::string> cellName;
  // Populated only for output ports that connect directly to input ports
  std::optional<std::string> connectedInputPort;

  Port()
      : type(PortType::INPUT),
        portName(""),
        cellName(std::nullopt),
        connectedInputPort(std::nullopt) {}

  Port(PortType type, std::string portName,
       std::optional<std::string> cellName = std::nullopt,
       std::optional<std::string> connectedInputPort = std::nullopt)
      : type(type),
        portName(std::move(portName)),
        cellName(std::move(cellName)),
        connectedInputPort(std::move(connectedInputPort)) {}
};

void registerFromAutoHogTranslation();

/// Translates the given operation from AutoHog.
OwningOpRef<Operation *> translateFromAutoHog(llvm::StringRef inputString,
                                              MLIRContext *context);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_SOURCE_AUTOHOG_AUTOHOGIMPORTER_H_
