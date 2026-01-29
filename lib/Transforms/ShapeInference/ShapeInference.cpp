#include "lib/Transforms/ShapeInference/ShapeInference.h"

#include <llvm/Support/Debug.h>

#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Verifier.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "shape-inference"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SHAPEINFERENCE
#include "lib/Transforms/ShapeInference/ShapeInference.h.inc"

namespace {
FailureOr<SmallVector<int64_t>> getShapeFromArrayAttr(ArrayAttr shapeAttr,
                                                      ShapedType shapedType,
                                                      Location loc) {
  SmallVector<int64_t> shape;
  for (auto attr : shapeAttr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      shape.emplace_back(intAttr.getInt());
    else
      return emitError(loc,
                       "Invalid shape attribute (expected ArrayAttr of "
                       "only IntegerAttr)");
  }

  if (shape.size() != shapedType.getRank())
    return emitError(
        loc, llvm::formatv("Shape annotation has wrong number of elements "
                           "(expected {0}, got {1}).",
                           shapedType.getRank(), shape.size()));
  return shape;
}

/// Helper function to handle RegionBranchOpInterface operations
// (e.g., scf.while, scf.if, affine.for)
void handleInterface(RegionBranchOpInterface regionBranchOp) {
  LLVM_DEBUG(llvm::dbgs() << "ShapeInference:\t\t\tcurrent op has "
                             "RegionBranchOpInterface.\n");

  // Get all successors that can be reached from the current op
  // (starting from current op is indicated by passing
  // RegionBranchPoint::parent() as the branch point. Successors can
  // be either regions, or "parent", i.e., the regionBranchOp itself
  SmallVector<RegionSuccessor> successors;

  // Note: the RegionBranchOpInterface allows ops to omit successors
  // that are not reachable, in which case unresolved type
  // mismatches might remain in those regions.
  regionBranchOp.getSuccessorRegions(RegionBranchPoint::parent(), successors);

  for (auto& successor : successors) {
    // We are interested in successor _regions_ here,
    // so ignore if the op has a control-flow edge loop to itself
    if (successor.isParent()) continue;

    LLVM_DEBUG(llvm::dbgs() << "ShapeInference:\t\t\tfound region successor: "
                            << successor.getSuccessor()->getLoc() << "\n");
    // We have a region successor, so we want to update
    // the block argument types of the successor
    // based on the types of the operands of the regionBranchOp
    // that will be forwarded to this region.

    // The "inputs" (block arguments) of the current successor
    // that are defined by the "forwarded" operands of the op
    auto successorInputs = regionBranchOp.getSuccessorInputs(successor);
    // Note: the block arguments might also contain "produced"
    // values (e.g., the index in a loop), the types of which
    // could technically also somehow depend on the updated
    // types of the other operands. However, this isn't the case
    // for the ops we're currently focused on (scf/affine)
    // and there doesn't seem to be a good way to handle this

    // The operands of the op that are forwarded to the successor
    auto succesorOperands = regionBranchOp.getEntrySuccessorOperands(successor);

    for (auto [input, operand] : llvm::zip(successorInputs, succesorOperands)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeInference:\t\t\tupdating successor input: " << input
                 << " with type of successor operand: " << operand << "\n");
      input.setType(operand.getType());
    }
  }
}

void handleInterface(RegionBranchTerminatorOpInterface terminator) {
  LLVM_DEBUG(llvm::dbgs() << "ShapeInference:\t\t\tcurrent op has "
                             "RegionBranchTerminatorOpInterface.\n");

  // All potential potential successors of the terminator
  // This can be the parent op or other regions (e.g., in scf.while)
  SmallVector<RegionSuccessor> successors;

  // the terminator interface doesn't have the
  // getSuccessorRegions/getSuccessorEntryRegions distinction.
  // This seems to be one of several oversights around this
  // interface when RegionBranchPoint was introduced.
  // https://github.com/llvm/llvm-project/commit/4dd744ac9c0f772a61dd91c84bc14d17e69aec51
  SmallVector<Attribute> operands(terminator->getNumOperands());
  terminator.getSuccessorRegions(operands, successors);

  // Get the parent RegionBranchOpInterface to access successor inputs
  auto parentOp = dyn_cast<RegionBranchOpInterface>(terminator->getParentOp());

  for (auto successor : successors) {
    // the terminator operands that are "passed" to the successor
    auto successorOperands = terminator.getSuccessorOperands(successor);

    // Special case: the "return/yield" back to the parent op
    if (successor.isParent()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeInference:\t\t\tfound parent successor\n");
      /// The RegionBranchTerminatorOpInterface requires that
      /// the successor operands and the parent's results match
      for (auto [operand, result] : llvm::zip(
               successorOperands, terminator->getParentOp()->getResults())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "ShapeInference:\t\t\tupdating parent result " << result
                   << " with type of successor operand " << operand << "\n");
        result.setType(operand.getType());
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "ShapeInference:\t\t\t"
                                 "found region successor: "
                              << successor.getSuccessor()->getLoc() << "\n");
      // Get the successor inputs from the parent operation - these are the
      // block arguments that need to be updated.
      if (parentOp) {
        auto successorInputs = parentOp.getSuccessorInputs(successor);
        for (auto [operand, input] :
             llvm::zip(successorOperands, successorInputs)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "ShapeInference:\t\t\tupdating successor input: "
                     << input << " with type of successor operand: " << operand
                     << "\n");
          input.setType(operand.getType());
        }
      }
    }
  }
}

}  // namespace

struct InferShape : public OpRewritePattern<func::FuncOp> {
  InferShape(MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter& rewriter) const override {
    /// Helper to determine whether any IR modifications are necessary
    bool changed = false;

    LLVM_DEBUG(llvm::dbgs()
               << "ShapeInference: running shape inference on function "
               << funcOp.getName() << "\n");

    /////////////////////////////////////////////////////////////////////
    //// Step 1: Resolve any shape-annotated args in Function Type   ////
    /////////////////////////////////////////////////////////////////////
    SmallVector<Type> inferredInputTypes;
    for (auto [idx, type] :
         llvm::enumerate(funcOp.getFunctionType().getInputs())) {
      auto shapedType = dyn_cast<ShapedType>(type);

      bool isDynamicShapedType = shapedType && !shapedType.hasStaticShape();
      auto shapeAttr = funcOp.getArgAttrOfType<ArrayAttr>(idx, "shape.shape");

      // Shape annotation on ShapedType type -> push back new type
      if (isDynamicShapedType && shapeAttr) {
        auto shape =
            getShapeFromArrayAttr(shapeAttr, shapedType, funcOp.getLoc());
        if (failed(shape)) return shape;
        // save the newly inferred argument type
        inferredInputTypes.emplace_back(
            shapedType.clone(shape.value(), shapedType.getElementType()));
        changed = true;
        continue;
      }

      // Dynamic ShapedType but no {shape.shape} annotation
      if (isDynamicShapedType && !shapeAttr) {
        emitWarning(funcOp.getLoc(),
                    "Found argument with dynamic shaped type but no shape "
                    "annotation.");
      }

      // Shape annotation on non-ShapedType type
      if (shapeAttr && !shapedType) {
        emitWarning(
            funcOp.getLoc(),
            "Ignoring shape annotation on argument with non-ShapedType type");
      }

      // Shape annotation on non-dynamic ShapedType type
      if (!isDynamicShapedType && shapeAttr) {
        emitWarning(funcOp.getLoc(),
                    "Ignoring shape annotation on argument with non-dynamic "
                    "shaped type");
      }

      inferredInputTypes.emplace_back(type);
    }

    /////////////////////////////////////////////////////////////////////
    //// Step 2: Walk the body, inferring and updating new types     ////
    /////////////////////////////////////////////////////////////////////

    // Only do this if there's been an annotated type and there's a body
    if (changed && !funcOp.isDeclaration()) {
      rewriter.startOpModification(funcOp);

      //// Update Block Argument types based on Annotations
      for (auto [type, arg] :
           llvm::zip(inferredInputTypes, funcOp.getArguments())) {
        arg.setType(type);
      }

      // WalkOrder::PreOrder is essential so that we can handle mapping
      // operands to block arguments for region-bearing ops before
      // the region itself is walked.
      auto walkResult =
          funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation* op) {
            LLVM_DEBUG(llvm::dbgs() << "ShapeInference:\t\twalking op: "
                                    << op->getName() << "\n");

            // for RegionBranchOps, we need to forward the types of their
            // operands to the block arguments of their successor regions
            if (auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op)) {
              // While some ops might have both RegionBranchOpInterface and
              // other interfaces, such as InferTypeOpInterface, we cannot rely
              // on those since the body and terminators have not yet been
              // updated and the op is therefore potentially invalid.
              // -> return early and skip other interface handling code
              handleInterface(regionBranchOp);
              return WalkResult::advance();
            }

            // We focus on operations with return type inference,
            // as these will likely complain if their operands' types change
            // but their return type hasn't been updated.
            // Technically, other ops could also cause issues,
            // For example, an op could allow only dynamically shaped tensors,
            // which means it would fail to verify after shape inference,
            // but failing on those is probably the right thing to do for the
            // compiler
            if (auto inferOp = dyn_cast<InferTypeOpInterface>(op)) {
              LLVM_DEBUG(llvm::dbgs() << "ShapeInference:\t\t\tcurrent op has "
                                         "InferTypeOpInterface.\n");
              SmallVector<Type> inferredReturnTypes;
              if (failed(inferOp.inferReturnTypes(
                      rewriter.getContext(), op->getLoc(), op->getOperands(),
                      op->getAttrDictionary(), op->getPropertiesStorage(),
                      op->getRegions(), inferredReturnTypes))) {
                emitError(op->getLoc(), "Failed to infer return types.");
                return WalkResult::interrupt();
              }

              assert(inferredReturnTypes.size() == op->getNumResults() &&
                     "Number of inferred return types must match the number of "
                     "results");

              for (size_t i = 0; i < inferredReturnTypes.size(); ++i) {
                if (inferredReturnTypes[i] == op->getResult(i).getType())
                  continue;
                op->getResult(i).setType(inferredReturnTypes[i]);
              }

              if (failed(verify(op))) {
                emitError(op->getLoc(),
                          "Operation failed to verify with its newly inferred "
                          "return type(s) after its operands' types were "
                          "updated during shape inference.");
                return WalkResult::interrupt();
              }
            }

            // If an operation is a (RegionBranch) terminator, we need to update
            // the return types of the results of the parent operation to match
            // the types of the terminators operands.
            if (auto terminator =
                    dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
              // Skip the func.return of the funcOp we're currently walking
              // since we handle that separately at the end of this pass
              if (terminator->getParentOp() == funcOp) {
                LLVM_DEBUG(llvm::dbgs()
                           << "ShapeInference:\t\t\t skipping terminator of "
                              "current function since function return types "
                              "are handled separately\n");

              } else {
                handleInterface(terminator);
              }
            }

            // TODO (#1784): support additional operations/interfaces

            return WalkResult::advance();
          });
      if (walkResult.wasInterrupted()) {
        rewriter.cancelOpModification(funcOp);
        return failure();
      }
    }

    /////////////////////////////////////////////////////////////////////
    ////  Step 3: Handle function return types                       ////
    /////////// /////////////////////////////////////////////////////////

    // Terminator return types (only if they exist, i.e., not a declaration)
    SmallVector<Type> terminatorOperandTypes;
    if (!funcOp.isDeclaration()) {
      terminatorOperandTypes = llvm::to_vector(funcOp.getBody()
                                                   .getBlocks()
                                                   .front()
                                                   .getTerminator()
                                                   ->getOperandTypes());

      if (terminatorOperandTypes.size() !=
          funcOp.getFunctionType().getResults().size())
        return emitError(funcOp.getLoc(),
                         "Mismatch between inferred number of returns from "
                         "function body and function signature.");
    }

    SmallVector<Type> inferredReturnTypes;

    for (auto [idx, t] : enumerate(funcOp.getFunctionType().getResults())) {
      auto shapedType = dyn_cast<ShapedType>(t);
      bool isDynamicShapedType = shapedType && !shapedType.hasStaticShape();
      auto shapeAttr = funcOp.getResultAttrOfType<ArrayAttr>(0, "shape.shape");

      if (isDynamicShapedType) {
        Type newType = Type();
        bool isAnnotated = false;
        if (shapeAttr) {
          auto shape =
              getShapeFromArrayAttr(shapeAttr, shapedType, funcOp.getLoc());
          if (failed(shape)) return shape;
          isAnnotated = true;
          newType =
              shapedType.clone(shape.value(), shapedType.getElementType());
        }
        if (!funcOp.isDeclaration()) {
          auto terminatorType = terminatorOperandTypes[idx];
          if (isAnnotated && terminatorType != newType) {
            return emitError(
                funcOp.getLoc(),
                "Mismatch between inferred return type and annotated "
                "return type.");
          }
          newType = terminatorType;
        }
        changed = true;
        inferredReturnTypes.emplace_back(newType);
      } else {
        // Dynamic ShapedType but no {shape.shape} annotation
        if (isDynamicShapedType && !shapeAttr) {
          emitWarning(funcOp.getLoc(),
                      "Found result with dynamic shaped type but no shape "
                      "annotation.");
        }

        // Shape annotation on non-ShapedType type
        if (shapeAttr && !shapedType) {
          emitWarning(
              funcOp.getLoc(),
              "Ignoring shape annotation on result with non-ShapedType type");
        }

        // Shape annotation on non-dynamic ShapedType type
        if (!isDynamicShapedType && shapeAttr) {
          emitWarning(funcOp.getLoc(),
                      "Ignoring shape annotation on result with non-dynamic "
                      "shaped type");
        }
        inferredReturnTypes.emplace_back(t);
      }
    }

    /////////////////////////////////////////////////////////////////////
    ////  Step 4: Update the function type                            ////
    /////////////////////////////////////////////////////////////////////

    // If no types changed, we don't need to do anything
    if (!changed) return failure();

    // If the function had no body, we did not yet signal a modification
    // Otherwise, we did so already in Step 2
    if (funcOp.isDeclaration()) rewriter.startOpModification(funcOp);

    // We need to update the function type to reflect the new types
    funcOp.setFunctionType(
        rewriter.getFunctionType(inferredInputTypes, inferredReturnTypes));

    // Also remove the attribute argument from the signature
    // (primarily for better readability of the IR after shape inference).
    // N.B.: these fail gracefully/silently if there's no such attribute
    for (int i = 0; i < inferredInputTypes.size(); ++i) {
      funcOp.removeArgAttr(i, "shape.shape");
    }
    for (int i = 0; i < inferredReturnTypes.size(); ++i) {
      funcOp.removeResultAttr(i, "shape.shape");
    }

    // Signal that we have successfully modified the op
    rewriter.finalizeOpModification(funcOp);
    return success();
  }
};

struct ShapeInference : impl::ShapeInferenceBase<ShapeInference> {
  using ShapeInferenceBase::ShapeInferenceBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<InferShape>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
