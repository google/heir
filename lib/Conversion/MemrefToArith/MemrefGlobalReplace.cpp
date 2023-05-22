#include <numeric>

#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h" // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h" // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h" // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h" // from @llvm-project

namespace mlir {
namespace heir {

class MemrefGlobalAsFuncPattern final : public mlir::ConversionPattern {
 public:
  explicit MemrefGlobalAsFuncPattern(mlir::MLIRContext *context)
      : mlir::ConversionPattern(mlir::memref::GlobalOp::getOperationName(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto global = dyn_cast<mlir::memref::GlobalOp>(op);
    auto memRefType = global.getType().cast<mlir::MemRefType>();
    auto resultElementType = memRefType.getElementType();

    // Ensure the global memref is a read-only constant.
    auto cstAttr =
        global.getConstantInitValue().dyn_cast_or_null<DenseElementsAttr>();
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
          getGlobal.emitError()
              << "MemrefGlobalLoweringPattern requires all global memref "
                 "readers to be affine reads, but got "
              << user;
          return mlir::failure();
        }

        // Get the affine load operations access indices.
        auto readOp = mlir::cast<affine::AffineReadOpInterface>(user);

        // Expand affine map from 'affineLoadOp'.
        affine::MemRefAccess readAccess(readOp);
        affine::AffineValueMap thisMap;
        readAccess.getAccessMap(&thisMap);
        mlir::SmallVector<uint64_t, 4> accessIndices;
        for (auto i = 0; i < readAccess.getRank(); ++i) {
          // The access indices of the global memref *must* be constant,
          // meaning that they cannot be a variable access (for example, a
          // loop index) or symbolic, for example, an input symbol.
          if (thisMap.getResult(i).getKind() != AffineExprKind::Constant) {
            readOp.emitError() << "MemrefGlobalLoweringPattern requires "
                                  "constant memref accessors";
            return mlir::failure();
          }
          accessIndices.push_back(
              (thisMap.getResult(i).dyn_cast<mlir::AffineConstantExpr>())
                  .getValue());
        }

        // Create an arithmetic constant from the global memref and
        // forward the load to this value.
        OpBuilder builder(readOp);
        auto val = *(constantAttrIt +
                     mlir::ElementsAttr::getFlattenedIndex(
                         cstAttr, llvm::ArrayRef<uint64_t>(accessIndices)));
        auto cst = builder.create<mlir::arith::ConstantOp>(
            user->getLoc(), resultElementType,
            mlir::cast<mlir::TypedAttr>(val));
        rewriter.replaceOp(readOp, {cst});
      }
      // Erase the get_global now that all its uses are replaced with
      // inline constants.
      rewriter.eraseOp(getGlobal);
    }

    // Erase the global after removing all of its users.
    rewriter.eraseOp(global);
    return mlir::success();
  }
};

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
    auto memRefType = global.getType().cast<mlir::MemRefType>();
    auto resultElementType = memRefType.getElementType();

    // Ensure the global memref is a read-only constant, so that we may
    // trivially forward its constant values to affine loads.
    auto cstAttr =
        global.getConstantInitValue().dyn_cast_or_null<DenseElementsAttr>();
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
          getGlobal.emitError()
              << "MemrefGlobalLoweringPattern requires all global memref "
                 "readers to be affine reads, but got "
              << user;
          return mlir::failure();
        }

        // Get the affine load operations access indices.
        auto readOp = mlir::cast<affine::AffineReadOpInterface>(user);

        // Expand affine map from 'affineLoadOp'.
        affine::MemRefAccess readAccess(readOp);
        affine::AffineValueMap thisMap;
        readAccess.getAccessMap(&thisMap);
        mlir::SmallVector<uint64_t, 4> accessIndices;
        for (auto i = 0; i < readAccess.getRank(); ++i) {
          // The access indices of the global memref *must* be constant,
          // meaning that they cannot be a variable access (for example, a
          // loop index) or symbolic, for example, an input symbol.
          if (thisMap.getResult(i).getKind() != AffineExprKind::Constant) {
            readOp.emitError() << "MemrefGlobalLoweringPattern requires "
                                  "constant memref accessors";
            return mlir::failure();
          }
          accessIndices.push_back(
              (thisMap.getResult(i).dyn_cast<mlir::AffineConstantExpr>())
                  .getValue());
        }

        // Create an arithmetic constant from the global memref and
        // forward the load to this value.
        OpBuilder builder(readOp);
        auto val = *(constantAttrIt +
                     mlir::ElementsAttr::getFlattenedIndex(
                         cstAttr, llvm::ArrayRef<uint64_t>(accessIndices)));
        auto cst = builder.create<mlir::arith::ConstantOp>(
            user->getLoc(), resultElementType,
            mlir::cast<mlir::TypedAttr>(val));
        rewriter.replaceOp(readOp, {cst});
      }
      // Erase the get_global now that all its uses are replaced with
      // inline constants.
      rewriter.eraseOp(getGlobal);
    }

    // Erase the global after removing all of its users.
    rewriter.eraseOp(global);
    return mlir::success();
  }
};

// MemrefGlobalReplacementPass forwards global memref constants loads to
// arithmetic constants.
struct MemrefGlobalReplacementPass
    : public mlir::PassWrapper<MemrefGlobalReplacementPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MemrefGlobalLoweringPattern>(&getContext());

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }

  mlir::StringRef getArgument() const final { return "memref-global-replace"; }
};

std::unique_ptr<Pass> createMemrefGlobalReplacePass() {
  return std::make_unique<MemrefGlobalReplacementPass>();
}

}  // namespace heir
}  // namespace mlir
