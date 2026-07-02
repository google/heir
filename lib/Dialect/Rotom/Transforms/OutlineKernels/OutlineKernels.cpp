#include "lib/Dialect/Rotom/Transforms/OutlineKernels/OutlineKernels.h"

#include <cstdint>
#include <optional>
#include <string>

#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/AttributeUtils.h"
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
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
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

bool isRotomElementwise(Operation* op) {
  if (!isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
           arith::MulFOp, arith::MulIOp>(op)) {
    return false;
  }
  // The kernel attribute is what marks this op as Rotom-lowered.
  return op->hasAttr(secret::SecretDialect::kKernelAttrName) &&
         op->getNumOperands() == 2 && op->getNumResults() == 1 &&
         isa<RankedTensorType>(op->getResult(0).getType());
}

struct OutlineKernels : impl::OutlineKernelsBase<OutlineKernels> {
  using OutlineKernelsBase::OutlineKernelsBase;

  // Builds (or reuses) the kernel function for `op` and replaces it with a
  // call. `kernelOperands` become the function arguments (tagged with
  // `operandLayouts`); `prologue` ops (e.g. a matmul's zero-init chain) are
  // cloned into the body ahead of `op`. Kernels are deduplicated by
  // (op name, layouts, operand/result types).
  void outlineOp(ModuleOp module, Operation* op, ValueRange kernelOperands,
                 ArrayRef<Attribute> operandLayouts, Attribute resultLayout,
                 ArrayRef<Operation*> prologue) {
    MLIRContext* ctx = module.getContext();
    SmallVector<Attribute> keyParts = {
        StringAttr::get(ctx, op->getName().getStringRef()),
        ArrayAttr::get(ctx, operandLayouts), resultLayout};
    for (Value operand : kernelOperands) {
      keyParts.push_back(TypeAttr::get(operand.getType()));
    }
    keyParts.push_back(TypeAttr::get(op->getResultTypes()[0]));
    Attribute key = ArrayAttr::get(ctx, keyParts);

    func::FuncOp& callee = outlinedKernels[key];
    if (!callee) {
      callee = buildKernel(module, op, kernelOperands, operandLayouts,
                           resultLayout, prologue);
    }

    OpBuilder builder(op);
    auto call =
        func::CallOp::create(builder, op->getLoc(), callee, kernelOperands);
    // The call result carries the kernel's result layout, so downstream
    // passes see the same per-value contract as before outlining.
    call->setAttr(kRotomLayoutAttrName, resultLayout);
    op->getResult(0).replaceAllUsesWith(call.getResult(0));
    op->erase();
    for (Operation* dead : llvm::reverse(prologue)) {
      if (dead->use_empty()) dead->erase();
    }
  }

  func::FuncOp buildKernel(ModuleOp module, Operation* op,
                           ValueRange kernelOperands,
                           ArrayRef<Attribute> operandLayouts,
                           Attribute resultLayout,
                           ArrayRef<Operation*> prologue) {
    MLIRContext* ctx = module.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // rotom_kernel_<mnemonic>_<n>, e.g. rotom_kernel_matmul_0.
    StringRef mnemonic = op->getName().getStringRef();
    mnemonic = mnemonic.drop_front(mnemonic.find('.') + 1);
    std::string name;
    do {
      name = ("rotom_kernel_" + mnemonic + "_" + Twine(kernelCounter++)).str();
    } while (module.lookupSymbol(name));

    SmallVector<Type> argTypes(kernelOperands.getTypes());
    auto funcType =
        builder.getFunctionType(argTypes, {op->getResultTypes()[0]});
    auto callee = func::FuncOp::create(builder, op->getLoc(), name, funcType);
    callee.setPrivate();
    callee->setAttr(kKernelFuncAttrName, UnitAttr::get(ctx));
    for (auto [index, layout] : llvm::enumerate(operandLayouts)) {
      callee.setArgAttr(index, kRotomLayoutAttrName, layout);
    }
    callee.setResultAttr(0, kRotomLayoutAttrName, resultLayout);

    Block* body = callee.addEntryBlock();
    builder.setInsertionPointToStart(body);
    IRMapping mapping;
    for (auto [index, operand] : llvm::enumerate(kernelOperands)) {
      mapping.map(operand, body->getArgument(index));
    }
    for (Operation* pre : prologue) builder.clone(*pre, mapping);
    // The clone keeps the op's rotom/kernel attributes, so the materializer
    // and the ciphertext lowering treat the body like any other
    // layout-assigned function.
    Operation* cloned = builder.clone(*op, mapping);
    func::ReturnOp::create(builder, op->getLoc(), cloned->getResult(0));
    return callee;
  }

  void outlineMatmul(ModuleOp module, linalg::MatmulOp op) {
    auto layouts = op->getAttrOfType<ArrayAttr>(kRotomMatmulAttrName);
    std::optional<MatmulInitChain> init = matchInitChain(op);
    if (!init) return;  // leave inline; the lowering handles it in place
    Operation* prologue[] = {init->fillValue, init->empty, init->fill};
    outlineOp(module, op, ValueRange{op.getInputs()[0], op.getInputs()[1]},
              {layouts[0], layouts[1]}, layouts[2], prologue);
  }

  void outlineElementwise(ModuleOp module, Operation* op) {
    FailureOr<Attribute> lhsLayout =
        findAttributeAssociatedWith(op->getOperand(0), kRotomLayoutAttrName);
    FailureOr<Attribute> rhsLayout =
        findAttributeAssociatedWith(op->getOperand(1), kRotomLayoutAttrName);
    Attribute resultLayout = op->getAttr(kRotomLayoutAttrName);
    if (failed(lhsLayout) || failed(rhsLayout) || !resultLayout) return;
    outlineOp(module, op, op->getOperands(), {*lhsLayout, *rhsLayout},
              resultLayout, /*prologue=*/{});
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation*> targets;
    module.walk([&](Operation* op) {
      auto parent = op->getParentOfType<func::FuncOp>();
      if (parent && parent->hasAttr(kKernelFuncAttrName)) return;
      // TODO: outline from inside secret.generic regions once the call /
      // region interaction is validated on the secret path.
      if (op->getParentOfType<secret::GenericOp>()) return;
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        if (matmul->getAttrOfType<ArrayAttr>(kRotomMatmulAttrName) &&
            matmul->getNumResults() == 1) {
          targets.push_back(op);
        }
        return;
      }
      if (isRotomElementwise(op)) targets.push_back(op);
    });
    for (Operation* op : targets) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        outlineMatmul(module, matmul);
      } else {
        outlineElementwise(module, op);
      }
    }
  }

  llvm::DenseMap<Attribute, func::FuncOp> outlinedKernels;
  int64_t kernelCounter = 0;
};

}  // namespace

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
