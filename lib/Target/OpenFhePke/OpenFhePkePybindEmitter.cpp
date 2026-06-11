#include "lib/Target/OpenFhePke/OpenFhePkePybindEmitter.h"

#include <cstddef>
#include <string>

#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

LogicalResult translateToOpenFhePkePybind(Operation* op, llvm::raw_ostream& os,
                                          const std::string& headerInclude,
                                          const std::string& pythonModuleName) {
  OpenFhePkePybindEmitter emitter(os, headerInclude, pythonModuleName);
  return emitter.translate(*op);
}

LogicalResult OpenFhePkePybindEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFhePkePybindEmitter::printOperation(ModuleOp moduleOp) {
  os << kPybindImports << "\n";
  os << "#include \"" << headerInclude_ << "\"\n";
  os << kPybindCommon << "\n";

  os << llvm::formatv(kPybindModuleTemplate.data(), pythonModuleName_) << "\n";
  os.indent();

  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult OpenFhePkePybindEmitter::printOperation(func::FuncOp funcOp) {
  llvm::StringRef funcName = canonicalizeDebugPort(funcOp.getName());
  if (funcName == "__heir_debug") {
    bool isVector = false;
    if (funcOp.getNumArguments() >= 3) {
      mlir::Type thirdArgType = funcOp.getArgumentTypes()[2];
      if (llvm::isa<ShapedType>(thirdArgType)) {
        isVector = true;
      }
    }

    if (isVector) {
      if (!boundHeirDebugVector_) {
        os << "m.def(\"__heir_debug\", py::overload_cast<CryptoContextT, "
              "PrivateKeyT, std::vector<CiphertextT>, const "
              "std::map<std::string, std::string>&>(&__heir_debug), "
              "py::call_guard<py::gil_scoped_release>());\n";
        boundHeirDebugVector_ = true;
      }
    } else {
      if (!boundHeirDebugSingle_) {
        os << "m.def(\"__heir_debug\", py::overload_cast<CryptoContextT, "
              "PrivateKeyT, CiphertextT, const std::map<std::string, "
              "std::string>&>(&__heir_debug), "
              "py::call_guard<py::gil_scoped_release>());\n";
        boundHeirDebugSingle_ = true;
      }
    }
  } else {
    // This branch is needed to support split preprocessing, which has multiple
    // return values.
    if (funcOp.getNumResults() > 1) {
      std::string structName = llvm::formatv("{0}Struct", funcName).str();
      os << llvm::formatv(kPybindStructClassTemplate.data(), structName);
      for (size_t i = 0; i < funcOp.getNumResults(); ++i) {
        os << llvm::formatv(kPybindStructFieldTemplate.data(), i, structName);
      }
      os << "    ;\n";
    }
    os << llvm::formatv(kPybindFunctionTemplate.data(), funcName) << "\n";
  }
  return success();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
