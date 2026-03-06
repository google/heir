#include "lib/Dialect/SCIFRBool/Transforms/ReplaceOpWithSection.h"

#include <cstdint>
#include <unordered_set>

#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolDialect.h"
#include "lib/Dialect/SCIFRBool/IR/SCIFRBoolOps.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project

// clang-format off
#include "lib/Dialect/SCIFRBool/Transforms/ReplaceOpWithSection.h.inc"
// clang-format on

namespace mlir {
namespace cornami {
namespace scifrbool {

#define GEN_PASS_DEF_REPLACEOPWITHSECTION
#include "lib/Dialect/SCIFRBool/Transforms/ReplaceOpWithSection.h.inc"

struct ReplaceOpWithSection
    : impl::ReplaceOpWithSectionBase<ReplaceOpWithSection> {
  using ReplaceOpWithSectionBase::ReplaceOpWithSectionBase;

  // TODO: Fuse Not op to the previous section greedily
  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext& context = getContext();

    op->walk([&](func::FuncOp op) { convertFunc(op, context); });
  }
  std::string getValueName(mlir::Value val) {
    std::string opName;
    llvm::raw_string_ostream ss(opName);
    mlir::OpPrintingFlags pFlags;

    val.printAsOperand(ss, pFlags);
    return opName;
  }

  void convertFunc(Operation* funcOp, MLIRContext& context) {
    uint32_t nChipsPerOp = 2;
    uint32_t nOpsPerSection = nChips / nChipsPerOp;
    std::vector<Operation*> opList;
    funcOp->walk([&](Operation* op) {
      if (isa<SCIFRBoolDialect>(op->getDialect()) && !isa<SectionOp>(op)) {
        opList.push_back(op);
      }
    });
    OpBuilder builder(&context);

    for (uint32_t i = 0; i < (uint32_t)opList.size(); i += nOpsPerSection) {
      auto startInsertionPoint = opList[i];
      // Collect the inputs and input types of the original operation
      SmallVector<Type, 4> inputTypes;
      SmallVector<Value, 4> inputTensors;
      std::unordered_set<std::string> inputNames;
      for (uint32_t j = 0; j < nOpsPerSection && i + j < opList.size(); j++) {
        Operation* scifrOp = opList[i + j];
        // Assuming the operation has operands (inputs)
        for (Value operand : scifrOp->getOperands()) {
          std::string opName = getValueName(operand);
          if (inputNames.find(opName) == inputNames.end()) {
            inputNames.insert(opName);
            inputTypes.push_back(operand.getType());
            // TODO: Do not add operands that will be created inside this
            // section as arguments
            inputTensors.push_back(operand);
          }
        }
      }
      builder.setInsertionPoint(startInsertionPoint);
      // Create a new SectionOp to wrap the original operator
      Operation* sectionOp = SectionOp::create(
          builder, startInsertionPoint->getLoc(), inputTypes[0], inputTensors);

      Region& region = sectionOp->getRegion(0);
      builder.createBlock(&region);
      Block* sectionOpBlock = &region.front();

      builder.setInsertionPointToStart(sectionOpBlock);
      for (uint32_t j = 0; j < nOpsPerSection && i + j < opList.size(); j++) {
        Operation* scifrOp = opList[i + j];
        Operation* scifrOpCopy = builder.clone(*scifrOp);
        scifrOpCopy->getBlock()->moveBefore(sectionOpBlock);
        scifrOp->replaceAllUsesWith(sectionOp->getResults());
        scifrOp->erase();
      }
    }
  }
};  // struct ReplaceOpWithSection

}  // namespace scifrbool
}  // namespace cornami
}  // namespace mlir
