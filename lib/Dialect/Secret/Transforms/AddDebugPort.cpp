#include "lib/Dialect/Secret/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "lib/Dialect/FuncUtils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"     // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETADDDEBUGPORT
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

func::FuncOp getOrCreateExternalDebugFunc(ModuleOp module, Type valueType) {
  std::string funcName = "__heir_debug_";

  SmallString<16> buffer;
  SmallString<16> buffer2;
  llvm::raw_svector_ostream os(buffer);
  valueType.print(os);
  funcName += sanitizeIdentifier(buffer, buffer2);

  auto* context = module.getContext();
  auto lookup = module.lookupSymbol<func::FuncOp>(funcName);
  if (lookup) return lookup;

  auto debugFuncType = FunctionType::get(context, {valueType}, {});

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  auto funcOp = func::FuncOp::create(b, funcName, debugFuncType);
  // required for external func call
  funcOp.setPrivate();
  return funcOp;
}

void insertValidationOps(secret::GenericOp op, DataFlowSolver& solver) {
  int count = 0;
  auto insertValidate = [&](Value value, OpBuilder& b) {
    if (isSecret(value, &solver)) {
      b.create<debug::ValidateOp>(value.getLoc(), value,
                                  "heir_debug_" + std::to_string(count++),
                                  nullptr);
    }
  };

  Block* body = op.getBody();
  OpBuilder argBuilder(body, body->begin());
  for (auto arg : body->getArguments()) {
    insertValidate(arg, argBuilder);
  }

  op.walk([&](Operation* walkOp) {
    if (walkOp == op.getOperation() || mlir::isa<secret::GenericOp>(walkOp) ||
        walkOp->hasTrait<OpTrait::IsTerminator>()) {
      return;
    }

    OpBuilder opBuilder(walkOp->getBlock(), ++walkOp->getIterator());
    for (Value result : walkOp->getResults()) {
      insertValidate(result, opBuilder);
    }
  });
}

void lowerValidationOps(secret::GenericOp op) {
  auto module = op->getParentOfType<ModuleOp>();
  op.walk([&](debug::ValidateOp validateOp) {
    Value value = validateOp.getInput();
    ImplicitLocOpBuilder b(validateOp.getLoc(), validateOp);

    auto callOp = b.create<func::CallOp>(
        getOrCreateExternalDebugFunc(module, value.getType()),
        ArrayRef<Value>{value});

    // Transfer attributes
    callOp->setAttr("debug.name", validateOp.getNameAttr());
    if (validateOp.getMetadata()) {
      callOp->setAttr("debug.metadata", validateOp.getMetadataAttr());
    }

    validateOp.erase();
  });
}

struct AddDebugPort : impl::SecretAddDebugPortBase<AddDebugPort> {
  using SecretAddDebugPortBase::SecretAddDebugPortBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());
    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](secret::GenericOp genericOp) {
      if (insertDebugAfterEveryOp) {
        insertValidationOps(genericOp, solver);
      }
      lowerValidationOps(genericOp);
    });
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
