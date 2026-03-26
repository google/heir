#include "lib/Target/OpenFhePke/OpenFhePkePybindEmitter.h"

#include <string>

#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

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
  os << llvm::formatv(kPybindFunctionTemplate.data(),
                      canonicalizeDebugPort(funcOp.getName()))
     << "\n";
  bool hasPrecomputeableOps = false;
  funcOp.walk([&](Operation* op) {
    auto name = op->getName().getStringRef();
    if (name == "openfhe.linear_transform") {
      hasPrecomputeableOps = true;
      return WalkResult::interrupt();
    }
    if (name == "openfhe.make_ckks_packed_plaintext") {
      if (isa<BlockArgument>(op->getOperand(1))) {
        hasPrecomputeableOps = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (hasPrecomputeableOps) {
    os << llvm::formatv(
              kPybindFunctionTemplate.data(),
              canonicalizeDebugPort(funcOp.getName()) + "__precompute")
       << "\n";
  }
  return success();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
