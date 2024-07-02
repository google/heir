#include "lib/Transforms/LazyRelin/LazyRelin.h"

namespace mlir{
namespace heir{

#define GEN_PASS_DEF_LAZYRELIN
#include "lib/Transforms/LazyRelin/LazyRelin.h.inc"

struct LazyRelin : impl::LazyRelinBase<LazyRelin>{
    using LazyRelinBase::LazyRelinBase;

    void runOnOperation() {
        Operation *module = getOperation();

        // TODO:
        // LazyRelinAnalysis analysis(module);
        
        
    }
}


}  // namespace heir
}  // namespace mlir