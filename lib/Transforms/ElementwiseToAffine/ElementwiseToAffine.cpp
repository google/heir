#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ELEMENTWISETOAFFINE
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h.inc"

// All of this is based on the ElementwiseToLinalg Pass in
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp

// An op is supported if it is ElementwiseMappable (all operands and results
// are mapped over), or if it has the ElementwiseByOperandOpInterface, which
// means only some operands are mapped over.
static bool isSupported(Operation* op) {
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    return llvm::any_of(op->getOperandTypes(),
                        [](Type type) { return isa<RankedTensorType>(type); });
  }

  if (auto interface = dyn_cast<ElementwiseByOperandOpInterface>(op)) {
    return llvm::all_of(
        llvm::enumerate(op->getOperands()), [&](auto indexedOperand) {
          return !interface.operandIsMappable(indexedOperand.index()) ||
                 isa<RankedTensorType>(indexedOperand.value().getType());
        });
  }

  return false;
}

namespace {

struct ConvertAnyElementwiseMappableOpOnRankedTensors : public RewritePattern {
  ConvertAnyElementwiseMappableOpOnRankedTensors(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const final {
    if (!isSupported(op))
      return rewriter.notifyMatchFailure(
          op,
          "unsupported op: not elementwise mappable, or doesn't have "
          "ElementwiseByOperandOpInterface, or doesn't have tensor operands");

    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "unsupported op: more than one result");

    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    Type elementType = resultType.getElementType();
    ArrayRef<int64_t> shape = resultType.getShape();
    int64_t rank = resultType.getRank();

    // Save insertion point prior to entering loop nest
    OpBuilder::InsertionGuard guard(rewriter);

    // Create an empty tensor as initial value of the iter_args
    Value target = tensor::EmptyOp::create(
        rewriter, op->getLoc(), shape, elementType, resultType.getEncoding());
    llvm::SmallVector<Value, 1> indices;

    // Create a an affine.for loop nest of depth rank
    for (int64_t i = 0; i < rank; ++i) {
      auto loop =
          affine::AffineForOp::create(rewriter, op->getLoc(), /* lowerBound*/ 0,
                                      /* upperBound*/ shape[i],
                                      /* step*/ 1,
                                      /* iterArgs*/ target);
      target = loop.getRegionIterArgs().front();
      indices.push_back(loop.getInductionVar());

      if (i == 0) {
        rewriter.replaceOp(op, loop);
      } else {
        affine::AffineYieldOp::create(rewriter, op->getLoc(),
                                      loop->getResults());
      }
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Create the innermost body
    auto resultTypes =
        llvm::to_vector<6>(llvm::map_range(op->getResultTypes(), [](Type type) {
          return cast<TensorType>(type).getElementType();
        }));

    // Generate a `tensor.extract` for each tensor operand
    SmallVector<Value, 4> newOperands;
    for (OpOperand& opOperand : op->getOpOperands()) {
      Value operand = opOperand.get();
      if (mlir::isa<TensorType>(operand.getType())) {
        if (OpTrait::hasElementwiseMappableTraits(op)) {
          // We don't need to check the shape, as ElementwiseMappable
          // requires all tensor operands to have compatible shapes
          auto extractOp = tensor::ExtractOp::create(rewriter, operand.getLoc(),
                                                     operand, indices);
          newOperands.push_back(extractOp);
          continue;
        }

        auto interface = cast<ElementwiseByOperandOpInterface>(op);
        int operandIndex = opOperand.getOperandNumber();
        if (interface.operandIsMappable(operandIndex)) {
          // We don't need to check the shape, as
          // ElementwiseByOperandOpInterface requires all mappable tensor
          // operands to have compatible shapes
          auto extractOp = tensor::ExtractOp::create(rewriter, operand.getLoc(),
                                                     operand, indices);
          newOperands.push_back(extractOp);
          continue;
        }
      }

      // We are left with either non-tensor operands (e.g., scalars) or tensor
      // operands which are not to be mapped over. These are replicated and
      // used as-is.
      newOperands.push_back(operand);
    }

    // "lowered" operation is the same operation, but over non-tensor
    // operands
    auto* scalarOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        newOperands, resultTypes, op->getAttrs());

    // insert scalarOp into the tensor at right index
    Value inserted = tensor::InsertOp::create(
        rewriter, op->getLoc(), scalarOp->getResult(0), target, indices);

    // replace lingalg.yield scalarOp with affine.yield insertedOp
    affine::AffineYieldOp::create(rewriter, op->getLoc(), inserted);
    return success();
  }
};
}  // namespace

struct ElementwiseToAffine
    : impl::ElementwiseToAffineBase<ElementwiseToAffine> {
  using ElementwiseToAffineBase::ElementwiseToAffineBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    auto walkResult = getOperation()->walk([&](Operation* op) {
      bool convertAll = convertDialects.empty() && convertOps.empty();
      bool convertDialect = llvm::is_contained(
          convertDialects, op->getDialect()->getNamespace().str());
      bool convertOp =
          llvm::is_contained(convertOps, op->getName().getStringRef().str());

      if (convertAll || convertDialect || convertOp) {
        if (isSupported(op)) {
          SmallVector<Type> operandAndResultTypes;
          for (Type ty : op->getOperandTypes())
            operandAndResultTypes.push_back(ty);
          for (Type ty : op->getResultTypes())
            operandAndResultTypes.push_back(ty);

          for (Type ty : operandAndResultTypes) {
            ShapedType shapedTy = dyn_cast<ShapedType>(ty);
            if (shapedTy && !shapedTy.hasStaticShape()) {
              op->emitError()
                  << "op has operand or result with dynamic shape, "
                     "which is not supported by elementwise-to-affine";
              return WalkResult::interrupt();
            }
          }
        }
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>(context);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      bool convertAll = convertDialects.empty() && convertOps.empty();
      bool convertDialect = llvm::is_contained(
          convertDialects, op->getDialect()->getNamespace().str());
      bool convertOp =
          llvm::is_contained(convertOps, op->getName().getStringRef().str());

      if (convertAll || convertDialect || convertOp) return !isSupported(op);
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
