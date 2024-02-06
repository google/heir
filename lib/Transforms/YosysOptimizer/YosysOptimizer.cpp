#include "include/Transforms/YosysOptimizer/YosysOptimizer.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretPatterns.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "lib/Transforms/YosysOptimizer/LUTImporter.h"
#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "yosys-optimizer"

namespace mlir {
namespace heir {
using std::string;

#define GEN_PASS_DEF_YOSYSOPTIMIZER
#include "include/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

// $0: verilog filename
// $1: function name
// $2: yosys runfiles
// $3: abc path
// $4: abc fast option -fast
constexpr std::string_view kYosysTemplate = R"(
read_verilog {0};
hierarchy -check -top \{1};
proc; memory; stat;
techmap -map {2}/techmap.v; stat;
opt; stat;
abc -exe {3} -lut 3 {4}; stat;
opt_clean -purge; stat;
rename -hide */c:*; rename -enumerate */c:*;
techmap -map {2}/map_lut_to_lut3.v; opt_clean -purge;
hierarchy -generate * o:Y i:*; opt; opt_clean -purge;
clean;
stat;
)";

struct YosysOptimizer : public impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;

  YosysOptimizer(std::string yosysFilesPath, std::string abcPath, bool abcFast,
                 int unrollFactor)
      : yosysFilesPath(std::move(yosysFilesPath)),
        abcPath(std::move(abcPath)),
        abcFast(abcFast),
        unrollFactor(unrollFactor) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override;

 private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
  int unrollFactor;
};

Value convertIntegerValue(Value value, Type convertedType, OpBuilder &b,
                          Location loc) {
  IntegerType argType = value.getType().cast<IntegerType>();
  int width = argType.getWidth();
  if (width == 1) {
    return value;
  }

  auto allocOp =
      b.create<memref::AllocOp>(loc, MemRefType::get({width}, b.getI1Type()));
  for (int i = 0; i < width; i++) {
    // These arith ops correspond to extracting the i-th bit
    // from the input
    auto shiftAmount =
        b.create<arith::ConstantOp>(loc, argType, b.getIntegerAttr(argType, i));
    auto bitMask = b.create<arith::ConstantOp>(
        loc, argType, b.getIntegerAttr(argType, 1 << i));
    auto andOp = b.create<arith::AndIOp>(loc, value, bitMask);
    auto shifted = b.create<arith::ShRSIOp>(loc, andOp, shiftAmount);
    b.create<memref::StoreOp>(
        loc, b.create<arith::TruncIOp>(loc, b.getI1Type(), shifted), allocOp,
        ValueRange{b.create<arith::ConstantIndexOp>(loc, i)});
  }

  return allocOp.getResult();
}

/// Convert a secret.generic's operands secret.secret<i3>
/// to secret.secret<memref<3xi1>>.
LogicalResult convertOpOperands(secret::GenericOp op, func::FuncOp func,
                                SmallVector<Value> &typeConvertedArgs) {
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type convertedType =
        func.getFunctionType().getInputs()[opOperand.getOperandNumber()];

    if (!opOperand.get().getType().isa<secret::SecretType>()) {
      // The type is not secret, but still must be booleanized
      OpBuilder builder(op);
      auto convertedValue = convertIntegerValue(opOperand.get(), convertedType,
                                                builder, op.getLoc());
      typeConvertedArgs.push_back(convertedValue);
      continue;
    }

    secret::SecretType originalType =
        opOperand.get().getType().cast<secret::SecretType>();

    if (!originalType.getValueType().isa<IntegerType, MemRefType>()) {
      op.emitError() << "Unsupported input type to secret.generic: "
                     << originalType.getValueType();
      return failure();
    }

    // Insert a conversion from the original type to the converted type
    OpBuilder builder(op);
    typeConvertedArgs.push_back(builder.create<secret::CastOp>(
        op.getLoc(), secret::SecretType::get(convertedType), opOperand.get()));
  }

  return success();
}

/// Convert a secret.generic's results from secret.secret<memref<3xi1>>
/// to secret.secret<i3>.
LogicalResult convertOpResults(secret::GenericOp op,
                               DenseSet<Operation *> &castOps,
                               SmallVector<Value> &typeConvertedResults) {
  for (Value opResult : op.getResults()) {
    // The secret.yield verifier ensures generic can only return secret types.
    assert(opResult.getType().isa<secret::SecretType>());
    secret::SecretType secretType =
        opResult.getType().cast<secret::SecretType>();

    IntegerType elementType;
    int numElements = 1;
    if (MemRefType convertedType =
            dyn_cast<MemRefType>(secretType.getValueType())) {
      if (!convertedType.getElementType().isa<IntegerType>() ||
          convertedType.getRank() != 1) {
        op.emitError() << "While booleanizing secret.generic, found converted "
                          "type that cannot be reassembled: "
                       << convertedType;
        return failure();
      }
      elementType = convertedType.getElementType().cast<IntegerType>();
      numElements = convertedType.getNumElements();
    } else {
      elementType = secretType.getValueType().cast<IntegerType>();
    }

    if (elementType.getWidth() != 1) {
      op.emitError() << "Converted element type must be i1";
      return failure();
    }

    IntegerType reassembledType =
        IntegerType::get(op.getContext(), elementType.getWidth() * numElements);

    // Insert a reassembly of the original integer type from its booleanized
    // memref version.
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto castOp = builder.create<secret::CastOp>(
        op.getLoc(), secret::SecretType::get(reassembledType), opResult);
    castOps.insert(castOp);
    typeConvertedResults.push_back(castOp.getOutput());
  }

  return success();
}

/// Move affine.apply to the start of an affine.for's body. This makes the
/// assumption that affine.apply's are independent of each other within the
/// body of a loop nest, which is probably not true in general, but may suffice
/// for this pass, in which the loop unrolling inserts a single affine.apply op
/// between two generics in the unrolled loop body.
class FrontloadAffineApply : public OpRewritePattern<affine::AffineApplyOp> {
 public:
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  FrontloadAffineApply(MLIRContext *context, affine::AffineForOp parentOp)
      : OpRewritePattern(context, /*benefit=*/3), parentOp(parentOp) {}

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto forOp = op->getParentOfType<affine::AffineForOp>();
    if (!forOp) return failure();
    if (forOp != parentOp) return failure();

    for (auto &earlierOp : forOp.getBody()->getOperations()) {
      if (&earlierOp == op) break;

      if (!isa<affine::AffineApplyOp>(earlierOp)) {
        rewriter.setInsertionPoint(&earlierOp);
        rewriter.replaceOp(op, rewriter.clone(*op.getOperation()));
        return success();
      }
    }
    return failure();
  }

 private:
  affine::AffineForOp parentOp;
};

/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class ExpandAffineApply : public OpRewritePattern<affine::AffineApplyOp> {
 public:
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap) return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

LogicalResult unrollAndMergeGenerics(Operation *op, int unrollFactor,
                                     DominanceInfo &domInfo,
                                     PostDominanceInfo &postDomInfo) {
  SmallVector<affine::AffineForOp> nestedLoops;

  auto walkResult =
      op->walk<WalkOrder::PreOrder>([&](affine::AffineForOp forOp) {
        LLVM_DEBUG(forOp.emitRemark() << "Visiting loop nest");
        SmallVector<affine::AffineForOp> nestedLoops;
        mlir::affine::getPerfectlyNestedLoops(nestedLoops, forOp);

        // We unroll the inner-most loop nest, if it consists of a single
        // generic as the body. Note that unrolling replaces the loop with a new
        // loop, and as a result the replaced loop is visisted again in the
        // walk. This means we must ensure that the post-condition of the
        // processing in this function. doesn't trigger this logic a second
        // time.
        affine::AffineForOp innerMostLoop = nestedLoops.back();
        // two ops because the last one must be affine.yield
        bool containsSingleOp =
            innerMostLoop.getBody()->getOperations().size() == 2;
        bool firstOpIsGeneric = isa<secret::GenericOp>(
            innerMostLoop.getBody()->getOperations().front());
        if (!containsSingleOp || !firstOpIsGeneric) {
          LLVM_DEBUG(innerMostLoop.emitRemark()
                     << "Skipping loop nest because either it contains more "
                     << "than one op or its sole op is not a generic op.\n");
          return WalkResult::skip();
        }

        if (failed(loopUnrollUpToFactor(innerMostLoop, unrollFactor))) {
          return WalkResult::interrupt();
        }
        LLVM_DEBUG(innerMostLoop.emitRemark() << "Post loop unroll");

        mlir::RewritePatternSet patterns(op->getContext());
        patterns.add<FrontloadAffineApply, secret::MergeAdjacentGenerics>(
            op->getContext(), innerMostLoop);
        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
          return WalkResult::interrupt();
        }

        LLVM_DEBUG(innerMostLoop.emitRemark()
                   << "Post merge generics (without cleanup)");
        // Success means we do not process any more nodes within this loop nest,
        // This corresponds to skipping this node in the walk.
        return WalkResult::skip();
      });

  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult runOnGenericOp(MLIRContext *context, secret::GenericOp op,
                             const std::string &yosysFilesPath,
                             const std::string &abcPath, bool abcFast) {
  std::string moduleName = "generic_body";

  // Only run this when there are arithmetic operations inside the generic body.
  auto result = op->walk([&](Operation *op) -> WalkResult {
    if (isa<arith::ArithDialect>(op->getDialect())) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!result.wasInterrupted()) return success();

  // Translate function to Verilog. Translation will fail if the func contains
  // unsupported operations.
  // TODO(#374): Directly convert MLIR to Yosys' AST instead of using Verilog.
  //
  // After that is done, it might make sense to rewrite this as a
  // RewritePattern, which only runs if the body does not contain any comb ops,
  // and generalize this to support converting a secret.generic as well as a
  // func.func. It's necessary to wait for the migration because the Yosys API
  // used here maintains global state that apparently does not play nicely with
  // the instantiation of multiple rewrite patterns.
  LLVM_DEBUG(op.emitRemark() << "Emitting verilog for this op");

  char *filename = std::tmpnam(nullptr);
  std::error_code ec;
  llvm::raw_fd_ostream of(filename, ec);
  if (failed(translateToVerilog(op, of, moduleName,
                                /*allowSecretOps=*/true)) ||
      ec) {
    op.emitError() << "Failed to translate to verilog";
    of.close();
    return failure();
  }
  of.close();

  LLVM_DEBUG({
    std::string result;
    llvm::raw_string_ostream os(result);
    [[maybe_unused]] auto res =
        translateToVerilog(op, os, moduleName, /*allowSecretOps=*/true);
    llvm::dbgs() << "Emitted verilog:\n" << os.str() << "\n";
  });

  // Invoke Yosys to translate to a combinational circuit and optimize.
  Yosys::log_error_stderr = true;
  LLVM_DEBUG(Yosys::log_streams.push_back(&std::cout));
  Yosys::run_pass(llvm::formatv(kYosysTemplate.data(), filename, moduleName,
                                yosysFilesPath, abcPath,
                                abcFast ? "-fast" : ""));

  // Translate Yosys result back to MLIR and insert into the func
  LLVM_DEBUG(Yosys::run_pass("dump;"));
  std::stringstream cellOrder;
  Yosys::log_streams.push_back(&cellOrder);
  Yosys::run_pass("torder -stop * P*;");
  Yosys::log_streams.clear();
  auto topologicalOrder = getTopologicalOrder(cellOrder);
  LUTImporter lutImporter = LUTImporter(context);
  Yosys::RTLIL::Design *design = Yosys::yosys_get_design();
  func::FuncOp func =
      lutImporter.importModule(design->top_module(), topologicalOrder);
  Yosys::run_pass("delete;");

  // The pass changes the yielded value types, e.g., from an i8 to a
  // memref<8xi1>. So the containing secret.generic needs to be updated and
  // conversions implemented on either side to convert the ints to memrefs
  // and back again.
  //
  // convertOpOperands goes from i8 -> memref<8xi1>
  // converOpResults from memref<8xi1> -> i8
  SmallVector<Value> typeConvertedArgs;
  typeConvertedArgs.reserve(op->getNumOperands());
  if (failed(convertOpOperands(op, func, typeConvertedArgs))) {
    return failure();
  }

  int resultIndex = 0;
  for (Type ty : func.getFunctionType().getResults())
    op->getResult(resultIndex++).setType(secret::SecretType::get(ty));

  // Replace the func.return with a secret.yield
  op.getRegion().takeBody(func.getBody());
  op.getOperation()->setOperands(typeConvertedArgs);

  Block &block = op.getRegion().getBlocks().front();
  func::ReturnOp returnOp = cast<func::ReturnOp>(block.getTerminator());
  OpBuilder bodyBuilder(&block, block.end());
  bodyBuilder.create<secret::YieldOp>(returnOp.getLoc(),
                                      returnOp.getOperands());
  returnOp.erase();
  func.erase();

  DenseSet<Operation *> castOps;
  SmallVector<Value> typeConvertedResults;
  castOps.reserve(op->getNumResults());
  typeConvertedResults.reserve(op->getNumResults());
  if (failed(convertOpResults(op, castOps, typeConvertedResults))) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Generic results: " << typeConvertedResults.size()
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Original results: " << op.getResults().size()
                          << "\n");

  op.getResults().replaceUsesWithIf(
      typeConvertedResults, [&](OpOperand &operand) {
        return !castOps.contains(operand.getOwner());
      });
  return success();
}

// Optimize the body of a secret.generic op.
void YosysOptimizer::runOnOperation() {
  Yosys::yosys_setup();
  auto *ctx = &getContext();
  auto *op = getOperation();

  if (unrollFactor > 0 && failed(unrollAndMergeGenerics(
                              op, unrollFactor, getAnalysis<DominanceInfo>(),
                              getAnalysis<PostDominanceInfo>()))) {
    signalPassFailure();
    return;
  }

  // Cleanup after unrollAndMergeGenerics
  mlir::RewritePatternSet cleanupPatterns(ctx);
  // We lift loads/stores into their own generics if possible, to avoid putting
  // the entire memref in the verilog module. Some loads would be hoistable but
  // they depend on arithmetic of index accessors that are otherwise secret.
  // Hence we need the HoistPlaintextOps provided by
  // populateGenericCanonicalizers in addition to special patterns that lift
  // loads and stores into their own generics.
  cleanupPatterns.add<secret::HoistOpBeforeGeneric>(
      op->getContext(), std::vector<std::string>{"memref.load", "affine.load"});
  cleanupPatterns.add<secret::HoistOpAfterGeneric>(
      op->getContext(),
      std::vector<std::string>{"memref.store", "affine.store"});
  secret::populateGenericCanonicalizers(cleanupPatterns, ctx);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(cleanupPatterns)))) {
    signalPassFailure();
    getOperation()->emitError()
        << "Failed to cleanup generic ops after unrollAndMergeGenerics";
    return;
  }

  // In general, a secret.generic pattern may not have all its ambient
  // plaintext variables passed through as inputs. The Yosys optimizer needs to
  // know all the inputs to the circuit, and capturing the ambient scope as
  // generic inputs is an easy way to do that.
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<secret::CaptureAmbientScope, secret::YieldStoredMemrefs>(ctx);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
    signalPassFailure();
    getOperation()->emitError()
        << "Failed to preprocess generic ops before yosys optimizer";
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "IR after cleanup in preparation for yosys optimizer\n";
    getOperation()->dump();
  });

  auto result = op->walk([&](secret::GenericOp op) {
    if (failed(runOnGenericOp(ctx, op, yosysFilesPath, abcPath, abcFast))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  Yosys::yosys_shutdown();

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createYosysOptimizer(
    const std::string &yosysFilesPath, const std::string &abcPath, bool abcFast,
    int unrollFactor) {
  return std::make_unique<YosysOptimizer>(yosysFilesPath, abcPath, abcFast,
                                          unrollFactor);
}

void registerYosysOptimizerPipeline(const std::string &yosysFilesPath,
                                    const std::string &abcPath) {
  PassPipelineRegistration<YosysOptimizerPipelineOptions>(
      "yosys-optimizer", "The yosys optimizer pipeline.",
      [yosysFilesPath, abcPath](OpPassManager &pm,
                                const YosysOptimizerPipelineOptions &options) {
        pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath,
                                        options.abcFast, options.unrollFactor));
        pm.addPass(mlir::createCSEPass());
      });
}

void registerUnrollAndOptimizeAnalysisPipeline(
    const std::string &yosysFilesPath, const std::string &abcPath) {
  PassPipelineRegistration<UnrollAndOptimizePipelineOptions>(
      "unroll-and-optimize-analysis",
      "An analysis tool for determining an optimal loop-unroll factor.",
      [yosysFilesPath, abcPath](
          OpPassManager &pm, const UnrollAndOptimizePipelineOptions &options) {
        for (int i = 0; i < 4; ++i) {
          pm.addPass(mlir::createCSEPass());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath,
                                          options.abcFast, /*unrollFactor=*/2));
          // TODO(#257): Implement statistics printer
        }
      });
}

}  // namespace heir
}  // namespace mlir
