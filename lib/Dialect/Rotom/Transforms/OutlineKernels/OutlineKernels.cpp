#include "lib/Dialect/Rotom/Transforms/OutlineKernels/OutlineKernels.h"

#include <cstdint>
#include <string>

#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/Twine.h"                 // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

#define GEN_PASS_DEF_OUTLINEKERNELS
#include "lib/Dialect/Rotom/Transforms/OutlineKernels/OutlineKernels.h.inc"

namespace {

constexpr llvm::StringLiteral kRotomLayoutAttrName = "rotom.layout";
// Marks an outlined kernel function so re-running the pass skips its body.
constexpr llvm::StringLiteral kKernelFuncAttrName = "rotom.kernel_func";

// The init chain the matmul lowering requires (and the only values a kernel
// body needs beyond its tensor operands): a zero fill of an empty tensor.
struct MatmulInitChain {
  arith::ConstantOp fillValue;
  tensor::EmptyOp empty;
  linalg::FillOp fill;
};

std::optional<MatmulInitChain> matchInitChain(linalg::MatmulOp op) {
  auto fill = op.getOutputs()[0].getDefiningOp<linalg::FillOp>();
  if (!fill) return std::nullopt;
  auto fillValue = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto empty = fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>();
  if (!fillValue || !empty) return std::nullopt;
  return MatmulInitChain{fillValue, empty, fill};
}

struct OutlineKernels : impl::OutlineKernelsBase<OutlineKernels> {
  using OutlineKernelsBase::OutlineKernelsBase;

  func::FuncOp buildMatmulKernel(ModuleOp module, linalg::MatmulOp op,
                                 ArrayAttr layouts,
                                 const MatmulInitChain& init) {
    MLIRContext* ctx = module.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    std::string name;
    do {
      name = ("rotom_kernel_matmul_" + Twine(kernelCounter++)).str();
    } while (module.lookupSymbol(name));

    auto funcType = builder.getFunctionType(
        {op.getInputs()[0].getType(), op.getInputs()[1].getType()},
        {op->getResultTypes()[0]});
    auto callee = func::FuncOp::create(builder, op.getLoc(), name, funcType);
    callee.setPrivate();
    callee->setAttr(kKernelFuncAttrName, UnitAttr::get(ctx));
    callee.setArgAttr(0, kRotomLayoutAttrName, layouts[0]);
    callee.setArgAttr(1, kRotomLayoutAttrName, layouts[1]);
    callee.setResultAttr(0, kRotomLayoutAttrName, layouts[2]);

    Block* body = callee.addEntryBlock();
    builder.setInsertionPointToStart(body);
    IRMapping mapping;
    mapping.map(op.getInputs()[0], body->getArgument(0));
    mapping.map(op.getInputs()[1], body->getArgument(1));
    builder.clone(*init.fillValue, mapping);
    builder.clone(*init.empty, mapping);
    builder.clone(*init.fill, mapping);
    // The clone keeps the op's rotom.layout / rotom.matmul attributes, so the
    // materializer and the ciphertext lowering treat the body like any other
    // layout-assigned function.
    Operation* cloned = builder.clone(*op, mapping);
    func::ReturnOp::create(builder, op.getLoc(), cloned->getResult(0));
    return callee;
  }

  void outlineMatmul(ModuleOp module, linalg::MatmulOp op) {
    auto layouts = op->getAttrOfType<ArrayAttr>(kRotomMatmulAttrName);
    std::optional<MatmulInitChain> init = matchInitChain(op);
    if (!init) return;  // leave inline; the lowering handles it in place

    // Two matmuls with the same layout combination and operand/result types
    // share one kernel function.
    MLIRContext* ctx = module.getContext();
    Attribute key = ArrayAttr::get(
        ctx, {StringAttr::get(ctx, op->getName().getStringRef()), layouts,
              TypeAttr::get(op.getInputs()[0].getType()),
              TypeAttr::get(op.getInputs()[1].getType()),
              TypeAttr::get(op->getResultTypes()[0])});
    func::FuncOp& callee = outlinedKernels[key];
    if (!callee) callee = buildMatmulKernel(module, op, layouts, *init);

    OpBuilder builder(op);
    auto call =
        func::CallOp::create(builder, op.getLoc(), callee,
                             ValueRange{op.getInputs()[0], op.getInputs()[1]});
    // The call result carries the kernel's result layout, so downstream
    // passes see the same per-value contract as before outlining.
    call->setAttr(kRotomLayoutAttrName, layouts[2]);
    op->getResult(0).replaceAllUsesWith(call.getResult(0));
    op->erase();
    linalg::FillOp fill = init->fill;
    tensor::EmptyOp empty = init->empty;
    arith::ConstantOp fillValue = init->fillValue;
    if (fill->use_empty()) fill->erase();
    if (empty->use_empty()) empty->erase();
    if (fillValue->use_empty()) fillValue->erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp op) {
      auto parent = op->getParentOfType<func::FuncOp>();
      if (parent && parent->hasAttr(kKernelFuncAttrName)) return;
      if (!op->getAttrOfType<ArrayAttr>(kRotomMatmulAttrName)) return;
      if (op->getNumResults() != 1) return;
      targets.push_back(op);
    });
    for (linalg::MatmulOp op : targets) outlineMatmul(module, op);
  }

  llvm::DenseMap<Attribute, func::FuncOp> outlinedKernels;
  int64_t kernelCounter = 0;
};

}  // namespace

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
