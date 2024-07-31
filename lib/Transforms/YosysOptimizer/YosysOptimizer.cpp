#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Target/Verilog/VerilogEmitter.h"
#include "lib/Transforms/YosysOptimizer/BooleanGateImporter.h"
#include "lib/Transforms/YosysOptimizer/LUTImporter.h"
#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/Statistic.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
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
#include "lib/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

// $0: verilog filename
// $1: function name
// $2: yosys runfiles
// $3: abc path
// $4: abc fast option -fast
// This template uses LUTs to optimize logic. It handles Verilog modules that
// may call submodules, utilizing splitnets to split output ports of the
// submodule into individual bits. Note that the splitnets command uses %n to
// target all submodules besides the main function.
constexpr std::string_view kYosysLutTemplate = R"(
read_verilog -sv {0};
hierarchy -check -top \{1};
proc; memory; stat;
techmap -map {2}/techmap.v; stat;
splitnets -ports \{1} %n;
flatten; opt_expr; opt; opt_clean -purge;
rename -hide */w:*; rename -enumerate */w:*;
abc -exe {3} -lut 3 {4}; stat;
opt_clean -purge; stat;
techmap -map {2}/map_lut_to_lut3.v; opt_clean -purge;
hierarchy -generate * o:Y i:*; opt; opt_clean -purge;
clean;
stat;
)";

// $0: verilog filename
// $1: function name
// $2: abc path
// $3: yosys runfiles path
// $4: abc fast option -fast
constexpr std::string_view kYosysBooleanTemplate = R"(
read_verilog {0};
hierarchy -check -top \{1};
proc; memory; stat;
techmap -map {3}/techmap.v; opt; stat;
abc -exe {2} -g AND,NAND,OR,NOR,XOR,XNOR {4};
opt_clean -purge; stat;
rename -hide */c:*; rename -enumerate */c:*;
hierarchy -generate * o:Y i:*; opt; opt_clean -purge;
clean;
stat;
)";

namespace {

int64_t countArithOps(Operation *op, ModuleOp moduleOp) {
  int64_t numArithOps = 0;
  auto isArithOp = [](Operation *op) -> bool {
    return isa<arith::ArithDialect>(op->getDialect()) &&
           !isa<arith::ConstantOp>(op);
  };

  op->walk([&](Operation *op) {
    if (isArithOp(op)) {
      numArithOps++;
    }
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
      numArithOps += countArithOps(funcOp, moduleOp);
    }
  });

  return numArithOps;
}

}  // namespace

struct RelativeOptimizationStatistics {
  std::string originalOp;
  int64_t numArithOps;
  int64_t numCells;
};

struct YosysOptimizer : public impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;

  YosysOptimizer(std::string yosysFilesPath, std::string abcPath, bool abcFast,
                 int unrollFactor, Mode mode, bool printStats)
      : yosysFilesPath(std::move(yosysFilesPath)),
        abcPath(std::move(abcPath)),
        abcFast(abcFast),
        printStats(printStats),
        unrollFactor(unrollFactor),
        mode(mode) {}

  void runOnOperation() override;

  LogicalResult runOnGenericOp(secret::GenericOp op);

 private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
  bool printStats;
  int unrollFactor;
  Mode mode;
  llvm::SmallVector<RelativeOptimizationStatistics> optStatistics;
};

Value convertIntegerValue(Value value, Type convertedType, OpBuilder &b,
                          Location loc) {
  IntegerType argType = mlir::cast<IntegerType>(value.getType());
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

    if (!mlir::isa<secret::SecretType>(opOperand.get().getType())) {
      // The type is not secret, but still must be booleanized
      OpBuilder builder(op);
      auto convertedValue = convertIntegerValue(opOperand.get(), convertedType,
                                                builder, op.getLoc());
      typeConvertedArgs.push_back(convertedValue);
      continue;
    }

    secret::SecretType originalType =
        mlir::cast<secret::SecretType>(opOperand.get().getType());

    if (!mlir::isa<IntegerType, MemRefType>(originalType.getValueType())) {
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
                               SmallVector<Type> originalResultTy,
                               DenseSet<Operation *> &castOps,
                               SmallVector<Value> &typeConvertedResults) {
  for (auto opResult : op->getResults()) {
    // The secret.yield verifier ensures generic can only return secret types.
    assert(mlir::isa<secret::SecretType>(opResult.getType()));
    secret::SecretType secretType =
        mlir::cast<secret::SecretType>(opResult.getType());

    IntegerType elementType;
    if (MemRefType convertedType =
            dyn_cast<MemRefType>(secretType.getValueType())) {
      if (!mlir::isa<IntegerType>(convertedType.getElementType()) ||
          convertedType.getRank() != 1) {
        op.emitError() << "While booleanizing secret.generic, found converted "
                          "type that cannot be reassembled: "
                       << convertedType;
        return failure();
      }
      elementType = mlir::cast<IntegerType>(convertedType.getElementType());
    } else {
      elementType = mlir::cast<IntegerType>(secretType.getValueType());
    }

    if (elementType.getWidth() != 1) {
      op.emitError() << "Converted element type must be i1";
      return failure();
    }

    // Insert a reassembly of the original integer type from its booleanized
    // memref version.
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto castOp = builder.create<secret::CastOp>(
        op.getLoc(), originalResultTy[opResult.getResultNumber()], opResult);
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
        // processing in this function doesn't trigger this logic a second
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
        LLVM_DEBUG(op->emitRemark() << "Post loop unroll");

        mlir::RewritePatternSet patterns(op->getContext());
        patterns.add<FrontloadAffineApply, secret::MergeAdjacentGenerics>(
            op->getContext(), innerMostLoop);
        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
          return WalkResult::interrupt();
        }

        LLVM_DEBUG(op->emitRemark() << "Post merge generics (without cleanup)");
        // Success means we do not process any more nodes within this loop nest,
        // This corresponds to skipping this node in the walk.
        return WalkResult::skip();
      });

  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult YosysOptimizer::runOnGenericOp(secret::GenericOp op) {
  std::string moduleName = "generic_body";
  MLIRContext *context = op->getContext();
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) return failure();

  // Count number of arith ops in the generic body
  int64_t numArithOps = countArithOps(op, moduleOp);
  if (numArithOps == 0) return success();

  optStatistics.push_back(RelativeOptimizationStatistics());
  auto &stats = optStatistics.back();
  if (printStats) {
    llvm::raw_string_ostream os(stats.originalOp);
    op->print(os);
    stats.numArithOps = numArithOps;
  }

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

  LLVM_DEBUG(
      llvm::dbgs() << "Using "
                   << (mode == Mode::LUT ? "LUT cells" : "boolean gates"));
  auto yosysTemplate =
      llvm::formatv(kYosysLutTemplate.data(), filename, moduleName,
                    yosysFilesPath, abcPath, abcFast ? "-fast" : "")
          .str();
  if (mode == Mode::Boolean) {
    yosysTemplate =
        llvm::formatv(kYosysBooleanTemplate.data(), filename, moduleName,
                      abcPath, yosysFilesPath, abcFast ? "-fast" : "")
            .str();
  }
  Yosys::run_pass(yosysTemplate);

  // Translate Yosys result back to MLIR and insert into the func
  LLVM_DEBUG(Yosys::run_pass("dump;"));
  std::stringstream cellOrder;
  Yosys::log_streams.push_back(&cellOrder);
  Yosys::run_pass("torder -stop * P*;");
  Yosys::log_streams.clear();
  auto topologicalOrder = getTopologicalOrder(cellOrder);
  Yosys::RTLIL::Design *design = Yosys::yosys_get_design();
  auto numCells = design->top_module()->cells().size();
  totalCircuitSize += numCells;
  if (printStats) {
    stats.numCells = numCells;
  }

  LLVM_DEBUG(llvm::dbgs() << "Importing RTLIL module\n");
  std::unique_ptr<RTLILImporter> importer;
  if (mode == Mode::LUT) {
    importer = std::make_unique<LUTImporter>(context);
  } else {
    importer = std::make_unique<BooleanGateImporter>(context);
  }
  func::FuncOp func = importer->importModule(
      design->top_module(), topologicalOrder,
      llvm::to_vector(llvm::map_range(op.getResultTypes(), [](Type ty) {
        return cast<secret::SecretType>(ty).getValueType();
      })));
  Yosys::run_pass("delete;");

  LLVM_DEBUG(llvm::dbgs() << "Done importing RTLIL, now type-coverting ops\n");

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

  SmallVector<Type> originalResultTypes;
  for (auto result : op->getResults()) {
    originalResultTypes.push_back(result.getType());
  }

  int resultIndex = 0;
  for (Type ty : func.getFunctionType().getResults())
    op->getResult(resultIndex++).setType(secret::SecretType::get(ty));

  DenseSet<Operation *> castOps;
  SmallVector<Value> typeConvertedResults;
  castOps.reserve(op->getNumResults());
  typeConvertedResults.reserve(op->getNumResults());
  if (failed(convertOpResults(op, originalResultTypes, castOps,
                              typeConvertedResults))) {
    return failure();
  }

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

  mlir::RewritePatternSet cleanupPatterns(ctx);
  if (unrollFactor > 1) {
    if (failed(unrollAndMergeGenerics(op, unrollFactor,
                                      getAnalysis<DominanceInfo>(),
                                      getAnalysis<PostDominanceInfo>()))) {
      signalPassFailure();
      return;
    }

    // Cleanup after unrollAndMergeGenerics
    // We lift loads/stores into their own generics if possible, to avoid
    // putting the entire memref in the verilog module. Some loads would be
    // hoistable but they depend on arithmetic of index accessors that are
    // otherwise secret. Hence we need the HoistPlaintextOps provided by
    // populateGenericCanonicalizers in addition to special patterns that lift
    // loads and stores into their own generics.
    cleanupPatterns.add<secret::HoistOpBeforeGeneric>(
        ctx, std::vector<std::string>{"memref.load", "affine.load"});
    cleanupPatterns.add<secret::HoistOpAfterGeneric>(
        ctx, std::vector<std::string>{"memref.store", "affine.store"});
  }

  secret::populateGenericCanonicalizers(cleanupPatterns, ctx);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(cleanupPatterns)))) {
    signalPassFailure();
    getOperation()->emitError() << "Failed to cleanup generic ops";
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

  mlir::IRRewriter builder(&getContext());
  auto result = op->walk([&](secret::GenericOp op) {
    // Now pass through any constants used after capturing the ambient scope.
    // This way Yosys can optimize constants away instead of treating them as
    // variables to the optimized body.
    genericAbsorbConstants(op, builder);

    if (failed(runOnGenericOp(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  Yosys::yosys_shutdown();

  if (printStats && !optStatistics.empty()) {
    for (auto &stats : optStatistics) {
      double ratio = (double)stats.numCells / stats.numArithOps;
      llvm::errs() << "Optimization stats for op: \n\n"
                   << stats.originalOp
                   << "\n\n  Starting arith op count: " << stats.numArithOps
                   << "\n  Ending cell count: " << stats.numCells
                   << "\n  Ratio: " << ratio << "\n\n";
    }
  }

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createYosysOptimizer(
    const std::string &yosysFilesPath, const std::string &abcPath, bool abcFast,
    int unrollFactor, Mode mode, bool printStats) {
  return std::make_unique<YosysOptimizer>(yosysFilesPath, abcPath, abcFast,
                                          unrollFactor, mode, printStats);
}

void registerYosysOptimizerPipeline(const std::string &yosysFilesPath,
                                    const std::string &abcPath) {
  PassPipelineRegistration<YosysOptimizerPipelineOptions>(
      "yosys-optimizer", "The yosys optimizer pipeline.",
      [yosysFilesPath, abcPath](OpPassManager &pm,
                                const YosysOptimizerPipelineOptions &options) {
        pm.addPass(createYosysOptimizer(yosysFilesPath, abcPath,
                                        options.abcFast, options.unrollFactor,
                                        options.mode, options.printStats));
        pm.addPass(mlir::createCSEPass());
      });
}

}  // namespace heir
}  // namespace mlir
