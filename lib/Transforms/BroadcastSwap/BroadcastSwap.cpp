#include "lib/Transforms/BroadcastSwap/BroadcastSwap.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <list>
#include <optional>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "broadcast-canonicalizations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_BROADCASTSWAP
#include "lib/Transforms/BroadcastSwap/BroadcastSwap.h.inc"

struct BroadcastChain {
  linalg::BroadcastOp broadcastOp;
  SmallVector<Operation*> elementWiseOps;
  SmallVector<Operation*> broadcastUsers;
  SmallVector<Operation*> elementWiseInputs;
};

struct SwapBroadcastAndElementWise : public OpRewritePattern<linalg::ReduceOp> {
 public:
  SwapBroadcastAndElementWise(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::ReduceOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  // Recursively move op and all its operand-defining ops before `target`
  static void moveOpAndDepsBefore(Operation* op, Operation* target,
                                  PatternRewriter& rewriter) {
    for (Value operand : op->getOperands()) {
      if (Operation* defOp = operand.getDefiningOp()) {
        // Only move if it's after target (i.e. would violate dominance)
        if (!defOp->isBeforeInBlock(target)) {
          moveOpAndDepsBefore(defOp, target, rewriter);
        }
      }
    }
    rewriter.moveOpBefore(op, target);
  }

  static bool isElementwiseOp(Operation* op) {
    if (auto linalgOp = dyn_cast<mlir::linalg::LinalgOp>(op)) {
      if (linalgOp.getNumReductionLoops() != 0) return false;

      for (auto map : linalgOp.getIndexingMapsArray()) {
        if (!map.isIdentity()) return false;
      }
      return true;
    }

    return op->hasTrait<mlir::OpTrait::Scalarizable>() ||
           op->hasTrait<mlir::OpTrait::Elementwise>();
  }

  static std::optional<BroadcastChain> getElementWiseOps(
      linalg::ReduceOp reduceOp) {
    SmallVector<Operation*> elementWiseOps = {};
    auto dims = reduceOp.getDimensions();

    std::list<Operation*> queue = {};
    for (auto user : reduceOp->getResult(0).getUsers()) {
      queue.push_back(user);
    }

    linalg::BroadcastOp broadcastOp = nullptr;
    SmallVector<Operation*> broadcastUsers = {};
    SmallVector<Operation*> elementWiseInputs = {};

    // BFS and try and find a broadcast op with the same dims
    while (!queue.empty()) {
      auto user = queue.front();
      queue.pop_front();

      LLVM_DEBUG(llvm::dbgs()
                 << "[swap-broadcast-and-elementwise] checking op: ");
      LLVM_DEBUG(user->print(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");

      if (llvm::is_contained(elementWiseOps, user)) {
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[swap-broadcast-and-elementwise] op was new\n");

      // check for a broadcast op with the same broadcast dimensions as the
      // reduce op
      if (auto bcastOp = dyn_cast<linalg::BroadcastOp>(user)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[swap-broadcast-and-elementwise] found broadcast op\n");
        // if we have multiple broadcast
        if (broadcastOp && broadcastOp != bcastOp) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[swap-broadcast-and-elementwise] Multiple broadcast "
                        "ops found in the chain\n");
          return std::nullopt;
        }
        broadcastOp = bcastOp;
        for (auto user : bcastOp.getOperation()->getUsers()) {
          broadcastUsers.push_back(user);
        }
        // print the broadcast dimensions
        LLVM_DEBUG(
            llvm::dbgs()
            << "[swap-broadcast-and-elementwise] broadcast dimensions: ");
        for (auto dim : broadcastOp.getDimensions()) {
          LLVM_DEBUG(llvm::dbgs() << dim << " ");
        }
        LLVM_DEBUG(llvm::dbgs() << "\n");

        // print the reduce dimensions
        LLVM_DEBUG(llvm::dbgs()
                   << "[swap-broadcast-and-elementwise] reduce dimensions: ");
        for (auto dim : dims) {
          LLVM_DEBUG(llvm::dbgs() << dim << " ");
        }
        LLVM_DEBUG(llvm::dbgs() << "\n");

        if (broadcastOp.getDimensions() == dims) {
          continue;
        } else {
          return std::nullopt;
        }
      }

      // check that the user is an elementwise op
      if (isElementwiseOp(user)) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "[swap-broadcast-and-elementwise] found element-wise op\n");
        elementWiseOps.push_back(user);
        for (auto operand : user->getOperands()) {
          if (!llvm::is_contained(elementWiseOps, operand.getDefiningOp()) &&
              !llvm::is_contained(elementWiseInputs, operand.getDefiningOp()) &&
              operand.getDefiningOp() != broadcastOp &&
              operand.getDefiningOp() != reduceOp) {
            elementWiseInputs.push_back(operand.getDefiningOp());
          }
        }

        // add all users of the elementwise op to the queue
        for (auto elementwiseUser : user->getUsers()) {
          queue.push_back(elementwiseUser);
        }
        // check if the op has 1 input (user) and the others are constants. If
        // not, return nullopt
        int numNonConstantOperands = 0;
        for (auto operand : user->getOperands()) {
          // skip the operand if it is used by other ops in the chain
          if (find(elementWiseOps, operand.getDefiningOp()) ||
              operand.getDefiningOp() == broadcastOp ||
              operand.getDefiningOp() == reduceOp) {
            continue;
          }
          if (!isa<arith::ConstantOp>(operand.getDefiningOp())) {
            numNonConstantOperands++;
          }
        }
        if (numNonConstantOperands > 1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[swap-broadcast-and-elementwise] Elementwise op found "
                        "with more than 1 non-constant operand\n");
          return std::nullopt;
        }
      } else {
        return std::nullopt;
      }
    }
    // no elements to swap with
    if (elementWiseOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[swap-broadcast-and-elementwise] No "
                                 "elementwise ops found in the chain\n");
      return std::nullopt;
    }

    // check that the elementwise ops are self contained
    for (auto elementwiseOp : elementWiseOps) {
      for (auto result : elementwiseOp->getResults()) {
        for (auto user : result.getUsers()) {
          if (!llvm::is_contained(elementWiseOps, user) &&
              user != broadcastOp) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[swap-broadcast-and-elementwise] Elementwise op "
                          "found with operand defined outside the chain\n");
            return std::nullopt;
          }
        }
      }
    }

    return {BroadcastChain{broadcastOp, elementWiseOps}};
  }

  LogicalResult matchAndRewrite(mlir::linalg::ReduceOp reduceOp,
                                PatternRewriter& rewriter) const override {
    if (llvm::cast<RankedTensorType>(reduceOp.getInputs()[0].getType())
            .getShape()
            .size() != 1) {
      return rewriter.notifyMatchFailure(
          reduceOp, "reduce op input has more than 1 dimension");
    }

    auto chain = getElementWiseOps(reduceOp);

    if (!chain) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[swap-broadcast-and-elementwise] No broadcast op found at "
                    "the end of the chain\n");
      return rewriter.notifyMatchFailure(
          reduceOp, "no broadcast op found at the end of the chain");
    }

    Value reduceOpResult = reduceOp.getResult(0);

    Value originalInit = chain->broadcastOp.getInit();
    if (Operation* defOp = originalInit.getDefiningOp()) {
      if (!defOp->isBeforeInBlock(reduceOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[swap-broadcast-and-elementwise] moving init op and its "
                      "dependencies before the reduce op: "
                   << originalInit << "\n");
        moveOpAndDepsBefore(defOp, reduceOp, rewriter);
      }
    }

    rewriter.setInsertionPointAfter(reduceOp);

    auto newBroadcast = linalg::BroadcastOp::create(
        rewriter, chain->broadcastOp.getLoc(), reduceOpResult, originalInit,
        chain->broadcastOp.getDimensions());

    // swap chain with broadcast
    reduceOpResult.replaceUsesWithIf(
        newBroadcast->getResult(0), [&](OpOperand& use) {
          return use.getOwner() != chain->broadcastOp &&
                 use.getOwner() != newBroadcast;
        });

    // create new constants for dense constants
    // fail otherwise
    for (auto op : chain->elementWiseOps) {
      for (auto operand : op->getOperands()) {
        // check if the operand is defined by an op in the chain
        if (llvm::is_contained(chain->elementWiseOps,
                               operand.getDefiningOp())) {
          continue;
        }
        if (operand != newBroadcast->getResult(0) &&
            operand != reduceOpResult) {
          if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
            auto oldAttr = llvm::cast<DenseElementsAttr>(constantOp.getValue());

            if (oldAttr.isSplat()) {
              // this can be easily expanded
              auto newType = RankedTensorType::get(
                  llvm::cast<RankedTensorType>(
                      newBroadcast->getResult(0).getType())
                      .getShape(),
                  oldAttr.getElementType());

              DenseElementsAttr newAttr = DenseElementsAttr::get(
                  newType, oldAttr.getSplatValue<Attribute>());
              auto newConstantOp = rewriter.create<arith::ConstantOp>(
                  constantOp.getLoc(), newType, newAttr);
              for (auto& use : op->getOpOperands()) {
                if (use.get() == constantOp.getResult()) {
                  use.set(newConstantOp.getResult());
                }
              }
            } else {
              return rewriter.notifyMatchFailure(
                  op,
                  "non-splat constant operand found that is not the reduce op "
                  "result or the broadcast result");
            }
          } else {
            return rewriter.notifyMatchFailure(
                op,
                "non-constant operand found that is not the reduce op result "
                "or the broadcast result");
          }
        }
      }
      // update op type to use the broadcast result type
      auto newType = newBroadcast->getResult(0).getType();
      if (op->getResult(0).getType() != newType) {
        rewriter.modifyOpInPlace(op,
                                 [&]() { op->getResult(0).setType(newType); });
      }
    }

    // update ops after the chain to use the last chain op instead of the
    // broadcast
    auto last_chain_op = chain->broadcastOp->getOperand(0).getDefiningOp();
    for (auto op : chain->broadcastUsers) {
      // skip the ops that are part of the chain
      if (op == newBroadcast || op == reduceOp ||
          llvm::is_contained(chain->elementWiseOps, op)) {
        continue;
      }
      // replace the old broadcast with the last chain op
      for (auto& operand : op->getOpOperands()) {
        if (operand.get() == chain->broadcastOp->getResult(0)) {
          operand.set(last_chain_op->getResult(0));
        }
      }
    }
    rewriter.replaceOp(chain->broadcastOp,
                       chain->elementWiseOps.back()->getResults());

    return success();
  }
};

struct BroadcastSwap : public impl::BroadcastSwapBase<BroadcastSwap> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet swapPatterns(context);
    swapPatterns.add<SwapBroadcastAndElementWise>(context);
    walkAndApplyPatterns(module, std::move(swapPatterns));
  }
};

}  // namespace heir
}  // namespace mlir
