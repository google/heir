#include "lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"       // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"    // from @llvm-project
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
#include "lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.h.inc"

Value createConstantFloat(ImplicitLocOpBuilder& b, double floatValue,
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
  return arith::ConstantOp::create(b, constantValuesAttr);
}

struct ConvertTosaSigmoid : public OpRewritePattern<mlir::tosa::SigmoidOp> {
 private:
  DataFlowSolver* solver;

 public:
  ConvertTosaSigmoid(DataFlowSolver* solver, mlir::MLIRContext* context)
      : OpRewritePattern<mlir::tosa::SigmoidOp>(context), solver(solver) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tosa::SigmoidOp op,
                                PatternRewriter& rewriter) const override {
    bool operandIsSecret = isSecret(op.getOperand(), solver);
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
        arith::MulFOp::create(b, coefficientDegreeOne, op.getOperand());
    auto calculateDegreeTwo =
        arith::MulFOp::create(b, op.getOperand(), op.getOperand());
    auto calculateDegreeThree =
        arith::MulFOp::create(b, calculateDegreeTwo, op.getOperand());
    auto coefficientMultiplyDegreeThree =
        arith::MulFOp::create(b, calculateDegreeThree, coefficientDegreeThree);

    auto sumDegreeZeroAndOne = arith::AddFOp::create(
        b, coefficientDegreeZero, coefficientMultiplyDegreeOne);
    auto totalSum = arith::AddFOp::create(b, sumDegreeZeroAndOne,
                                          coefficientMultiplyDegreeThree);
    rewriter.replaceOp(op, totalSum);
    return success();
  }
};

struct TosaToSecretArith
    : public impl::TosaToSecretArithBase<TosaToSecretArith> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
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
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace tosa
}  // namespace heir
}  // namespace mlir
