#include "lib/Dialect/Preprocessing/Transforms/UseDynamicResourceDir.h"

#include <cassert>

#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

// IWYU pragma: begin_keep
#include "llvm/include/llvm/ADT/SetVector.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_USEDYNAMICRESOURCEDIR
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

namespace {

struct UseDynamicResourceDir
    : impl::UseDynamicResourceDirBase<UseDynamicResourceDir> {
  using UseDynamicResourceDirBase::UseDynamicResourceDirBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = &getContext();
    SymbolTable symbolTable(module);

    SetVector<func::FuncOp> needsDirectory;
    DenseMap<func::FuncOp, SmallVector<func::CallOp, 4>> calleeToCalls;

    // Single walk: collect resource loaders (and validate parent function)
    // as well as indexing calls by callee.
    WalkResult walkResult = module.walk([&](Operation* op) {
      if (auto load = dyn_cast<LoadResourceOp>(op)) {
        if (!load.getDirectory()) {
          auto func = load->getParentOfType<func::FuncOp>();
          if (!func) {
            load.emitOpError("expected to be contained within a func.func");
            return WalkResult::interrupt();
          }
          needsDirectory.insert(func);
        }
      } else if (auto call = dyn_cast<func::CallOp>(op)) {
        func::FuncOp callee =
            symbolTable.lookup<func::FuncOp>(call.getCallee());
        if (callee) {
          calleeToCalls[callee].push_back(call);
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) return signalPassFailure();

    // Transitively close over callers using a worklist over calleeToCalls.
    SmallVector<func::FuncOp> worklist(needsDirectory.begin(),
                                       needsDirectory.end());
    while (!worklist.empty()) {
      func::FuncOp target = worklist.pop_back_val();
      auto it = calleeToCalls.find(target);
      if (it != calleeToCalls.end()) {
        for (func::CallOp call : it->second) {
          if (auto caller = call->getParentOfType<func::FuncOp>()) {
            if (needsDirectory.insert(caller)) {
              worklist.push_back(caller);
            }
          }
        }
      }
    }

    Type directoryType = ResourceDirType::get(ctx);
    DenseMap<func::FuncOp, Value> funcToDirectory;

    for (func::FuncOp function : needsDirectory) {
      assert(!function.isDeclaration() &&
             "functions in needsDirectory must have bodies");

      // Check if function already has a resource_dir argument.
      Value directory = nullptr;
      for (Value arg : function.getArguments()) {
        if (isa<ResourceDirType>(arg.getType())) {
          directory = arg;
          break;
        }
      }

      if (!directory) {
        unsigned index = function.getNumArguments();
        if (failed(function.insertArgument(index, directoryType,
                                           DictionaryAttr::get(ctx),
                                           function.getLoc()))) {
          return signalPassFailure();
        }
        directory = function.getArgument(index);
      }
      funcToDirectory[function] = directory;

      function.walk([&](LoadResourceOp load) {
        if (load->getParentOfType<func::FuncOp>() == function &&
            !load.getDirectory()) {
          load.getDirectoryMutable().assign(directory);
        }
      });
    }

    // Forward the resource directory at all call sites of affected functions.
    for (func::FuncOp target : needsDirectory) {
      auto it = calleeToCalls.find(target);
      if (it == calleeToCalls.end()) continue;

      for (func::CallOp call : it->second) {
        auto caller = call->getParentOfType<func::FuncOp>();
        if (!caller) {
          call.emitOpError(
              "calls a resource-loading function from outside a "
              "function, so no resource directory reaches it");
          return signalPassFailure();
        }
        Value directory = funcToDirectory.lookup(caller);
        if (!directory) {
          call.emitOpError(
              "caller function does not have a resource directory argument");
          return signalPassFailure();
        }

        // Only append if the call does not already pass a resource directory
        // argument.
        bool alreadyHasDir =
            !call.getOperands().empty() &&
            isa<ResourceDirType>(call.getOperands().back().getType());
        if (call.getNumOperands() < target.getNumArguments() &&
            !alreadyHasDir) {
          call->insertOperands(call.getNumOperands(), directory);
          if (ArrayAttr argAttrs = call.getArgAttrsAttr()) {
            SmallVector<Attribute> newArgAttrs(argAttrs.begin(),
                                               argAttrs.end());
            newArgAttrs.push_back(DictionaryAttr::get(ctx));
            call.setArgAttrsAttr(ArrayAttr::get(ctx, newArgAttrs));
          }
        }
      }
    }
  }
};

}  // namespace

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
