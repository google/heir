#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define CheddarOps

#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"

// Generated definitions
#include "lib/Dialect/Cheddar/IR/CheddarDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarOps.cpp.inc"

namespace mlir {
namespace heir {
namespace cheddar {

namespace {
// Allow the inliner to clone cheddar ops. Without an inliner interface the
// MLIR inliner treats every cheddar op as illegal-to-inline, which blocks
// inlining any client function that contains them (e.g. folding the
// per-`func` preprocessing/compute decomposition back into a combined entry
// point). Cheddar ops carry no nested regions or control flow, so cloning is
// always safe.
struct CheddarInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
    return true;
  }
  bool isLegalToInline(Region*, Region*, bool, IRMapping&) const final {
    return true;
  }
};
}  // namespace

void CheddarDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Cheddar/IR/CheddarTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Cheddar/IR/CheddarOps.cpp.inc"
      >();

  addInterfaces<CheddarInlinerInterface>();
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
