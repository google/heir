#include "lib/Conversion/TosaToSecretArith/TosaToSecretArith.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "tosa-to-secret-arith"

namespace mlir {
namespace heir {
namespace tosa {

#define GEN_PASS_DEF_TOSATOSECRETARITH
#include "lib/Conversion/TosaToSecretArith/TosaToSecretArith.h.inc"

Value createConstantFloat(ImplicitLocOpBuilder &b, double floatValue,
                          RankedTensorType type) {
  auto elementType = type.getElementType();

  // Create APFloat based on the float type width
  APFloat value(0.0);  // Default initialization
  if (elementType.isF32()) {
    value = APFloat(static_cast<float>(
        floatValue));  // Convert double to float if necessary
  } else if (elementType.isF64()) {
    value = APFloat(floatValue);  // Use the double value directly
  } else {
    llvm_unreachable("Expected a valid float type for constant creation");
  }

  auto constantValuesAttr = SplatElementsAttr::get(type, value);
  return b.create<arith::ConstantOp>(constantValuesAttr);
}

struct ConvertTosaSigmoid : public OpRewritePattern<mlir::tosa::SigmoidOp> {
 private:
  DataFlowSolver *solver;

 public:
  ConvertTosaSigmoid(DataFlowSolver *solver, mlir::MLIRContext *context)
      : OpRewritePattern<mlir::tosa::SigmoidOp>(context), solver(solver) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tosa::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    auto isSecret = [&](Value value) {
      auto *operandLookup = solver->lookupState<SecretnessLattice>(value);
      Secretness operandSecretness =
          operandLookup ? operandLookup->getValue() : Secretness();
      return (operandSecretness.isInitialized() &&
              operandSecretness.getSecretness());
    };

    // Do not support lowering for non-secret operands
    bool operandIsSecret = isSecret(op.getOperand());
    if (!operandIsSecret) {
      return failure();
    }

    auto inputTensorType =
        dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!inputTensorType) {
      return failure();
    }

    auto dimensions = inputTensorType.getShape();
    auto dataType = inputTensorType.getElementType();

    // Do not support lowering for non-float types
    if (!dyn_cast<FloatType>(dataType)) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Calculates -0.004 * x^3 + 0.197 * x + 0.5
    auto rankedTensorType = RankedTensorType::get(dimensions, dataType);
    auto coefficientDegreeZero = createConstantFloat(b, 0.5, rankedTensorType);
    auto coefficientDegreeOne = createConstantFloat(b, 0.197, rankedTensorType);
    auto coefficientDegreeThree =
        createConstantFloat(b, -0.004, rankedTensorType);

    auto coefficientMultiplyDegreeOne =
        b.create<arith::MulFOp>(coefficientDegreeOne, op.getOperand());
    auto calculateDegreeTwo =
        b.create<arith::MulFOp>(op.getOperand(), op.getOperand());
    auto calculateDegreeThree =
        b.create<arith::MulFOp>(calculateDegreeTwo, op.getOperand());
    auto coefficientMultiplyDegreeThree =
        b.create<arith::MulFOp>(calculateDegreeThree, coefficientDegreeThree);

    auto sumDegreeZeroAndOne = b.create<arith::AddFOp>(
        coefficientDegreeZero, coefficientMultiplyDegreeOne);
    auto totalSum = b.create<arith::AddFOp>(sumDegreeZeroAndOne,
                                            coefficientMultiplyDegreeThree);
    rewriter.replaceOp(op, totalSum);
    return success();
  }
};

struct TosaToSecretArith
    : public impl::TosaToSecretArithBase<TosaToSecretArith> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(module);

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);

    patterns.add<ConvertTosaSigmoid>(&solver, context);

    // Run pattern matching and conversion
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace tosa
}  // namespace heir
}  // namespace mlir
