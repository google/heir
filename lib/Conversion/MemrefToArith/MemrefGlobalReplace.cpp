#include <cassert>
#include <memory>
#include <utility>

#include "lib/Conversion/MemrefToArith/MemrefToArith.h"
#include "lib/Conversion/MemrefToArith/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"         // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "memref-global-replace"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_MEMREFGLOBALREPLACEPASS
#include "lib/Conversion/MemrefToArith/MemrefToArith.h.inc"

// MemrefGlobalLoweringPattern lowers global memrefs by looking for its usages
// in modules and replacing them with in-module memref allocations and stores.
// In order for all memref.globals to be lowered, this pattern requires that
// loops are unrolled (to provide constant indices on any reads) and that all
// memref aliasing and copy operations are expanded.
class MemrefGlobalLoweringPattern final : public mlir::ConversionPattern {
 public:
  explicit MemrefGlobalLoweringPattern(mlir::MLIRContext *context)
      : mlir::ConversionPattern(mlir::memref::GlobalOp::getOperationName(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto global = dyn_cast<mlir::memref::GlobalOp>(op);
    auto memRefType = mlir::cast<mlir::MemRefType>(global.getType());
    auto resultElementType = memRefType.getElementType();

    // Ensure the global memref is a read-only constant, so that we may
    // trivially forward its constant values to affine loads.
    auto cstAttr = mlir::dyn_cast_or_null<DenseElementsAttr>(
        global.getConstantInitValue());
    if (!cstAttr) {
      op->emitError(
          "MemrefGlobalLoweringPattern requires global memrefs are read-only "
          "dense-valued constants");
      return mlir::failure();
    }
    auto constantAttrIt = cstAttr.value_begin<mlir::Attribute>();

    // Traverse the parent region for uses of the global memref.
    auto blockUse = SymbolTable::getSymbolUses(global.getSymNameAttr(),
                                               global->getParentRegion());
    bool getGlobalRemoveable = true;
    for (const auto &use : *blockUse) {
      auto getGlobal = mlir::cast<mlir::memref::GetGlobalOp>(use.getUser());
      assert(getGlobal);

      auto memrefUsers = getGlobal.getResult().getUsers();
      for (auto *user : memrefUsers) {
        // Require all users are affine readers. While some memref.load
        // ops could be supported if the index inputs are statically known,
        // for now requiring affine reads (and the precondition that loops
        // are unrolled) are sufficient to ensure we can statically
        // compute the load's input index.
        if (!isa<affine::AffineReadOpInterface>(user)) {
          LLVM_DEBUG(
              getGlobal.emitRemark()
              << "MemrefGlobalLoweringPattern requires all global memref "
                 "readers to be affine reads, but got "
              << user);
          getGlobalRemoveable = false;
          continue;
        }

        // Get the affine load operations access indices.
        auto readOp = mlir::cast<affine::AffineReadOpInterface>(user);
        affine::MemRefAccess readAccess(readOp);

        // Expand affine map from 'affineLoadOp'.
        auto flattenedIndex =
            getFlattenedAccessIndex(readAccess, readOp.getMemRefType());
        if (!flattenedIndex.has_value()) {
          LLVM_DEBUG(readOp->emitRemark()
                     << "MemrefGlobalLoweringPattern requires "
                        "constant memref accessors");
          getGlobalRemoveable = false;
          continue;
        }

        // Create an arithmetic constant from the global memref and
        // forward the load to this value.
        OpBuilder builder(readOp);
        auto val = *(constantAttrIt + flattenedIndex.value());
        auto cst = builder.create<mlir::arith::ConstantOp>(
            user->getLoc(), resultElementType,
            mlir::cast<mlir::TypedAttr>(val));
        rewriter.replaceOp(readOp, cst);
      }

      // Erase the get_global now that all its uses are replaced with
      // inline constants.
      if (getGlobalRemoveable) {
        rewriter.eraseOp(getGlobal);
      }
    }

    // Erase the global after removing all of its users.
    if (getGlobalRemoveable) {
      rewriter.eraseOp(global);
    } else {
      return rewriter.notifyMatchFailure(op, "could not replace global");
    }
    return mlir::success();
  }
};

// MemrefGlobalReplacementPass forwards global memref constants loads to
// arithmetic constants.
struct MemrefGlobalReplacementPass
    : impl::MemrefGlobalReplacePassBase<MemrefGlobalReplacementPass> {
  using MemrefGlobalReplacePassBase::MemrefGlobalReplacePassBase;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MemrefGlobalLoweringPattern>(&getContext());

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
