#include "lib/Transforms/LazyRelin/LazyRelin.h"

#include "lib/Analysis/LazyRelinAnalysis/LazyRelinAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir{
namespace heir{

#define GEN_PASS_DEF_LAZYRELIN
#include "lib/Transforms/LazyRelin/LazyRelin.h.inc"

struct LazyRelin : impl::LazyRelinBase<LazyRelin>{
    using LazyRelinBase::LazyRelinBase;

    void runOnOperation() {
        Operation *module = getOperation();

        LazyRelinAnalysis analysis(module);
        OpBuilder b(&getContext());

        module->walk([&](Operation *op){
            if(!analysis.shouldInsertRelin(op))
                return;

            b.setInsertionPointAfter(op);
            auto reduceOp = b.create<bgv::Relinearize>(op->getLoc(), op->getResult(0));
            op->getResult(0).replaceAllUsesExcept(reduceOp.getResult(), {reduceOp});
        
        });
        
        DataFlowSolver solver;
        
        solver.load<dataflow::DeadCodeAnalysis>();
        solver.load<dataflow::IntegerRangeAnalysis>();

        if(failed(solver.initializeAndRun(module))){
            getOperation()->emitOpError() << "Failed to run the analysis.\n";
            signalPassFailure();
            return;
        }
        
        auto result = module->walk([&](Operation *op){
            if(!llvm::isa<bgv::MulOp>(*op)){
                return WalkResult::advance();
            }
            const dataflow::IntegerValueRangeLattice *opRange = 
                solver.lookupState<dataflow::IntegerValueRangeLattice>(
                    op->getResult(0));

            if(!opRange || opRange->getValue().isUninitialized()){
                op->emitOpError()
                    << "Found op without a set integer range; did the analysis fail?";
                return WalkResult::interrupt();
            }

            ConstantIntRanges range = opRange->getValue().getValue();
            if(range.umax().getZExtValue() > 3){
                op->emitOpError() << "Found op after which the noise exceeds the "
                             "allowable maximum of "
                          << 3 //MAX_degree
                          << "; it was: " << range.umax().getZExtValue()
                          << "\n";
                    return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if(result.wasInterrupted()){
            getOperation()->emitOpError()
                << "Detected error in the noise analysis.\n";
            signalPassFailure();
        }
    }
};

}  // namespace heir
}  // namespace mlir