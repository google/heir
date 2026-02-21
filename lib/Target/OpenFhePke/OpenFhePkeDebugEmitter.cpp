#include "lib/Target/OpenFhePke/OpenFhePkeDebugEmitter.h"

#include <string>

#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

LogicalResult translateToOpenFhePkeDebugEmitter(
    Operation* op, llvm::raw_ostream& os, OpenfheImportType importType,
    const std::string& debugImportPath) {
  OpenFhePkeDebugEmitter emitter(os, importType, debugImportPath);
  return emitter.translate(*op);
}

LogicalResult OpenFhePkeDebugEmitter::translate(Operation& op) {
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

LogicalResult OpenFhePkeDebugEmitter::printOperation(ModuleOp moduleOp) {
  OpenfheScheme scheme;
  if (moduleIsBGV(moduleOp)) {
    scheme = OpenfheScheme::BGV;
  } else if (moduleIsBFV(moduleOp)) {
    scheme = OpenfheScheme::BFV;
  } else if (moduleIsCKKS(moduleOp)) {
    scheme = OpenfheScheme::CKKS;
  } else {
    return emitError(moduleOp.getLoc(), "Missing scheme attribute on module");
  }

  if (!debugImportPath.empty()) {
    os << "#include \"" << debugImportPath << "\"\n";
  }

  os << KdebugHeaderImports << "\n";
  os << "#include <iostream>" << "\n";
  os << getModulePrelude(scheme, importType_) << "\n";

  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult OpenFhePkeDebugEmitter::emitDebugHelperImpl() {
  os << "auto " << kIsBlockArgVar << " = " << kDebugAttrMapParam
     << ".at(\"asm.is_block_arg\");\n";

  os << llvm::formatv("if ({0} == \"1\") {{\n", kIsBlockArgVar);
  os.indent();
  os << "std::cout << \"Input\" << std::endl;\n";
  os.unindent();
  os << "}";
  os << llvm::formatv(" else {{\n");
  os.indent();
  os << "std::cout" << " " << "<< ";
  os << kDebugAttrMapParam << ".at" << "(\"asm.op_name\") << std::endl;\n";
  os.unindent();
  os << "}\n";
  os << "\n";

  os << "PlaintextT " << kPlaintxtVar << ";\n";
  os << kCctxtVar << "->Decrypt(" << kPrivKeyTVar << ", " << kCiphertxtVar
     << ", &" << kPlaintxtVar << ");\n";
  os << kPlaintxtVar << "->SetLength(std::stod(" << kDebugAttrMapParam
     << ".at(\"message.size\")));\n";
  os << "std::cout << \"  \" << " << kPlaintxtVar << " << std::endl;\n";
  return success();
}

LogicalResult OpenFhePkeDebugEmitter::printOperation(func::FuncOp funcOp) {
  if ((!isDebugPort(funcOp.getName())) || isEmitted) {
    return success();
  }

  auto res = emitDebugHelperSignature(
      funcOp, os, [&](Location loc, const std::string& message) {
        return emitError(loc, message);
      });

  if (failed(res)) {
    return res;
  }

  os << " {\n";
  os.indent();
  res = emitDebugHelperImpl();
  if (failed(res)) {
    return res;
  }
  os.unindent();
  os << "}\n";
  isEmitted = true;
  return success();
}

OpenFhePkeDebugEmitter::OpenFhePkeDebugEmitter(
    raw_ostream& os, OpenfheImportType importType,
    const std::string& debugImportPath)
    : importType_(importType),
      os(os),
      debugImportPath(debugImportPath),
      isEmitted(false) {}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
