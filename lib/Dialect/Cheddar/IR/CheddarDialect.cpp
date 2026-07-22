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
// Cheddar setup/keygen ops interact with stateful handles and (secret) key
// material rather than being pure value computations: create_context and
// create_user_interface mint distinct objects, prepare_rot_key generates a
// rotation key as a side effect, and encrypt/decrypt touch the user interface.
// Duplicating any of these into multiple call sites would create divergent
// contexts or redundantly (and observably) regenerate keys, so they must not be
// *cloned*. The pure ciphertext-algebra ops have value semantics and are always
// safe to clone.
bool isStatefulHandleOp(Operation* op) {
  return isa<CreateContextOp, CreateUserInterfaceOp, PrepareRotKeyOp, EncryptOp,
             DecryptOp>(op);
}

// Lets the inliner fold client functions that contain cheddar ops (e.g. the
// per-`func` preprocessing/compute decomposition back into a combined entry
// point). Without an interface the inliner treats every cheddar op as
// illegal-to-inline and blocks all such inlining.
struct CheddarInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // An op may be inlined as long as we are not duplicating a stateful op:
  // moving (wouldBeCloned == false) preserves single execution and order and is
  // always fine; cloning is only safe for the pure ciphertext-algebra ops.
  bool isLegalToInline(Operation* op, Region*, bool wouldBeCloned,
                       IRMapping&) const final {
    return !wouldBeCloned || !isStatefulHandleOp(op);
  }
  // Cheddar ops carry no nested regions or control flow, so a callee body is
  // structurally inlinable; per-op cloning safety is enforced above.
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
