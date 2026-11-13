#include "lib/Dialect/Polynomial/Transforms/PolyMulToNTT.h"

#include <cstdint>

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Dialect/Polynomial/Transforms/NTTSolver.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/EquivalenceClasses.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep
#include "mlir/include/mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "ortools/sat/cp_model.h"  // from @com_google_ortools

namespace mlir {
namespace heir {
namespace polynomial {

using operations_research::sat::BoolVar;

#define GEN_PASS_DEF_POLYMULTONTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

enum class OpFormClass {
  // This may be a monotypic class, with ReturnOp as a special case
  // In short, it means that the input can be in either form. It may
  // be possible to handles this as a special case inside "EITHER"
  // for ops that have no poly outputs, but I'm keeping it separate
  // for now.
  RETURN,
  // Ops in this class require:
  //  - all polynomial inputs MUST be in coeff form
  //  - all polynomial outputs MUST be in coeff form
  COEFF,
  // Ops in this class require:
  //  - all polynomial inputs MUST be in eval form
  //  - all polynomial outputs MUST be in eval form
  EVAL,
  // Ops in this class can either work in "coeff mode" or "eval mode".
  // When operating in "<X> mode":
  //  - all polynomial inputs MUST be in <X> form
  //  - all polynomial outputs MUST be in <X> form
  EITHER,
  // Ops in this class should be considered "precomputable constants"
  // meaning the are available in either form, or both forms, for free.
  CONST,
  // A class for otherwise-unclassified ops that result in an
  // error in this pass.
  UNKNOWN
};

OpFormClass opFormClass(Operation* op) {
  if (isa<func::ReturnOp>(op)) {
    return OpFormClass::RETURN;
  } else if (isa<ToTensorOp, LeadingTermOp, EvalOp, MonomialOp,
                 MonicMonomialMulOp, FromTensorOp, ApplyCoefficientwiseOp>(
                 op)) {
    return OpFormClass::COEFF;
  } else if (isa<MulOp>(op)) {
    return OpFormClass::EVAL;
  } else if (isa<AddOp, SubOp, MulScalarOp, ModSwitchOp, ExtractSliceOp,
                 tensor::ExtractSliceOp, tensor::ExtractOp,
                 tensor::FromElementsOp>(op)) {
    return OpFormClass::EITHER;
  } else if (isa<ConstantOp>(op)) {
    return OpFormClass::CONST;
  }
  return OpFormClass::UNKNOWN;
}

struct PolyMulToNTT : public impl::PolyMulToNTTBase<PolyMulToNTT> {
  using PolyMulToNTTBase::PolyMulToNTTBase;

  void runOnOperation() override;
};

static bool isPolyType(Type t) {
  if (auto p = dyn_cast<PolynomialType>(t)) return true;
  auto rt = dyn_cast<RankedTensorType>(t);
  if (rt && dyn_cast<PolynomialType>(rt.getElementType())) {
    return true;
  }
  return false;
}

static bool isPolyValue(Value v) { return isPolyType(v.getType()); }

static llvm::SmallVector<Value> filterPolynomialOps(ValueRange values) {
  llvm::SmallVector<Value> result;
  for (Value v : values) {
    if (isPolyValue(v)) {
      result.push_back(v);
    }
  }
  return result;
}

// Adds solver constraints for a RegionBranchOpInterface op (scf.for, scf.if,
// scf.while, ...): every successor input can be fed by more than one
// operand, and one op can have more than one independent
// loop-carried slot (e.g. two unrelated scf.for iter_args), each of which
// needs its own, independently-shared mode. This one call covers loop
// entry, backedges, and results, without needing to special-case any
// individual op.
// Every polynomial successor input encountered is added to
// `polySuccessorInputs`, so that Step 3b can materialize it once solving is
// done.
static void addRegionBranchConstraints(
    NTTSolver& solver, RegionBranchOpInterface regionBranchOp,
    llvm::SetVector<Value>& polySuccessorInputs) {
  RegionBranchSuccessorMapping operandToInputs;
  // A note on terminology, using scf.for as an example:
  // %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) ->
  // (!poly_ty) {
  //  %next = polynomial.mul %acc, %acc : !poly_ty
  //  scf.yield %next : !poly_ty
  // }
  //
  // %init and %next are operands forwarded across a region boundary. %acc and
  // %result are the "successor inputs" that receive them -- %acc is the body
  // region's successor input (a block argument), %result is the parent's
  // successor input (an op result). A "successor" is the *destination*
  // itself (the region, or "the parent"), not a value.
  //
  // getSuccessorOperandInputMapping returns a map from each such operand to
  // the successor input(s) it feeds, e.g. {%init: [%acc, %result], %next:
  // [%acc, %result]} (both fan out to both here, since the trip count isn't
  // statically known in this example).
  //
  // %acc and %result must resolve to the exact same materialized MLIR type --
  // MLIR requires a loop's iter_arg types to match its result types -- so they
  // share a single NTTSolver::addFloatingMode mode variable (the same
  // mechanism used for a single "either form" op's output, generalized to a
  // set of values instead of just one). Grouping them this way, rather than
  // tying a "native form" bit by hard equality, is what lets each successor
  // input keep its own, independently-forced demand (e.g. %result may be
  // coeff-only because of a downstream to_tensor, while %acc is eval-only
  // because of an eval-only op in the loop body) while still agreeing on one
  // shared, cost-optimal materialized form.
  regionBranchOp.getSuccessorOperandInputMapping(operandToInputs);

  // Groups the successor inputs into their independent loop-carried slots
  // (e.g. an scf.for with two unrelated iter_args produces two disjoint
  // groups).
  llvm::EquivalenceClasses<Value> groups;
  llvm::DenseMap<Value, llvm::SmallVector<Value>> targetToSources;
  for (auto& [operand, inputs] : operandToInputs) {
    Value source = operand->get();
    if (!isPolyValue(source)) continue;

    for (Value target : inputs) {
      polySuccessorInputs.insert(target);
      targetToSources[target].push_back(source);
      // One operand can feed more than one target -- e.g. scf.for's yielded
      // value feeds both the next iter_arg and the loop's result -- so they
      // belong in the same group.
      groups.unionSets(inputs.front(), target);
    }
  }

  for (const auto* group : groups) {
    if (!group->isLeader()) continue;
    llvm::SmallVector<Value> members(groups.member_begin(*group),
                                     groups.member_end());
    BoolVar mode = solver.addFloatingMode(members);
    for (Value target : members) {
      for (Value source : targetToSources[target]) {
        // Whichever form the group's shared mode picks, source (the operand
        // feeding target) must supply -- e.g. if the group picks EVAL, the
        // yielded value must supply EVAL too. If target additionally needs
        // the other form, that's handled below by addModeConversionCost as
        // a separate, locally materialized conversion at target's own
        // definition site, without requiring source to supply it.
        solver.implyMode(target, source);
      }
      solver.addModeConversionCost(target, mode);
    }
  }
}

void PolyMulToNTT::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* context = &getContext();
  NTTSolver solver;

  if (func.isDeclaration()) {
    // I'm returning a failure here because this approach doesn't properly
    // handle declarations. After updating the function definition, we need to
    // *find* any declarations and make them match.
    signalPassFailure();
    return;
  }

  IRRewriter rewriter(context);
  (void)runRegionDCE(rewriter, getOperation()->getRegions());
  RewritePatternSet canonicalizationPatterns(context);
  if (failed(
          applyPatternsGreedily(func, std::move(canonicalizationPatterns)))) {
    signalPassFailure();
    return;
  }
  (void)runRegionDCE(rewriter, getOperation()->getRegions());

  // Our goal is to insert as few NTTs + iNTTs as possible while satisyfing all
  // op constraints. We optimize at the function level, which means there should
  // be no NTTs/INTTs on inputs (unless both forms are needed) and no NTTs/INTTs
  // on outputs. Instead, we allow inputs and outputs to be in either form, and
  // choose whatever is "naturally" falls out.
  // This is optimal because the function we are optimizing does no
  // unnecessary transformations. Because of this constraint in particular,
  // max-flow/min-cut is not an option: it cannot express, e.g., "accept this
  // input in whatever form is best". Instead, we set up a (binary) constraint
  // satisfaction problem (essentially a subset of a generic integer linear
  // program) and find an optimal solution that minimizes the number of
  // NTTs+INTTs.
  //
  // At a high level, the approach is to split each value in the function into a
  // coeff form variable and an eval form variable. We add appropriate
  // constraints between inputs and outputs, and between the forms themselves,
  // and then solve the CP-SAT problem. Any "conversion" variables that are 1
  // correspond to places we need to insert an NTT or INTT.
  //
  // In more detail, pick any MLIR Value v in the input AST. We create several
  // (binary) variables corresponding to v:
  //   - v_c = 1 iff some consumer of v requires v in coeff form
  //   - v_e = 1 iff some consumer of v requires v in eval form
  //   - v_conv = 1 iff we require an NTT/INTT on the value v
  //   - v_mode is only for Values output by ops that can work in either form
  //     It is 0 if the SAT instance chooses to run the op on coeff-form values,
  //     and 1 if the SAT instance chooses to run the op on eval-form values.
  //
  // We proceed in five steps:
  //  1. Build the CP-SAT instance
  //  2. Solve the CP-SAT instance
  //  3. Use the CP-SAT solution to fix the *output* of ops in the AST. This
  //     introduces any needed conversions, but leaves inputs unchanged.
  //  4. Now that all necessary ops exist, walk the tree one final time to
  //     fix up the inputs
  //  5. Fix the function signature and arguments
  //
  // Ops implementing RegionBranchOpInterface (scf.for, scf.if, scf.while,
  // ...) are supported via a separate mechanism layered on top of the five
  // steps above; see addRegionBranchConstraints and Steps 3b/4b for details.
  // A conversion's cost is weighted by how many times it actually runs
  // (see getConversionCostMultiplier in NTTSolver.cpp): a loop with a static
  // trip count (e.g. scf.for with constant lower/upper bound and step)
  // contributes its exact iteration count, and any other loop is assumed to
  // run many times, so the solver strongly prefers hoisting a conversion out
  // of it.
  //
  // A block argument or region-branch result that isn't fed by any operand
  // (e.g. a loop's induction variable) is not a forwarding edge and is not
  // supported; this isn't a real limitation in practice since such
  // "produced" successor inputs are never polynomial-typed for the ops this
  // pass deals with.

  // Steps 1, 3, and 4 above involve walking the AST. Since we're going to be
  // doing multiple walks and adding some nodes on the way, we first memoize
  // the AST (so that we're not walking and mutating at the same time) and
  // prune it to remove ops that don't involve polynomials. This doesn't remove
  // ops from the AST, it just means that we don't walk over them later.
  llvm::SmallVector<Operation*> rewriteOrder;
  // RegionBranchOpInterface ops (e.g. scf.for, scf.if, scf.while) collected
  // here have their polynomial operands/results handled entirely by the
  // dedicated region-branch forwarding logic below (see
  // addRegionBranchConstraints), not by the generic per-op walks over
  // rewriteOrder.
  llvm::SmallVector<RegionBranchOpInterface> regionBranchOps;
  WalkResult wr =
      func.walk([&](Operation* op) -> WalkResult {
        // RegionBranchTerminatorOpInterface ops that actually terminate a
        // region of a RegionBranchOpInterface parent (e.g. scf.yield inside
        // scf.for/scf.if, scf.condition inside scf.while) never need their
        // own entry in rewriteOrder: every polynomial operand they have is a
        // forwarding edge into some successor input, captured via the
        // parent's successor mapping and rewritten directly through that
        // mapping's OpOperand pointers.
        //
        // The interface check alone isn't enough to identify these: many
        // ReturnLike ops (e.g. func.return, polynomial.yield) satisfy
        // RegionBranchTerminatorOpInterface "for free" regardless of what
        // their parent op is, even when that parent has nothing to do with
        // region-branching (e.g. func.func, polynomial.apply_coefficientwise)
        // and must still be handled by the ordinary per-op walks below. So we
        // additionally require the parent op to actually implement
        // RegionBranchOpInterface.
        Operation* parentOp = op->getParentOp();
        if (isa<RegionBranchTerminatorOpInterface>(op) && parentOp &&
            isa<RegionBranchOpInterface>(parentOp)) {
          return WalkResult::advance();
        }
        if (auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op)) {
          // Likewise, a region branch op's own operands/results (e.g.
          // scf.for's initial iter_arg operand, or a value returned to the
          // parent) are forwarding edges, not ordinary op inputs/outputs, so
          // this op is excluded from the single-poly-result restriction
          // below: each of its successor inputs gets its own, independently
          // solved form, so there's no bound on how many polynomial
          // loop-carried values it may have.
          if (!filterPolynomialOps(op->getOperands()).empty() ||
              !filterPolynomialOps(op->getResults()).empty()) {
            regionBranchOps.push_back(regionBranchOp);
          }
          return WalkResult::advance();
        }

        auto polyResults = filterPolynomialOps(op->getResults());
        auto polyOperands = filterPolynomialOps(op->getOperands());

        if (!polyOperands.empty() || !polyResults.empty()) {
          rewriteOrder.push_back(op);
          if (polyResults.size() > 1) {
            op->emitOpError()
                << "Walk 1: CP-SAT instance is only set up to support "
                   "ops that have at most one output, but "
                << op->getName() << " has " << polyResults.size() << " outputs";
            signalPassFailure();
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
  if (wr.wasInterrupted()) return;

  /***************************************************
   ********** Step 1: Build CP-SAT instance **********
   **************************************************/

  for (Value arg : func.getArguments()) {
    if (isPolyValue(arg)) {
      // Function arguments have no producing op to dictate a native form, so
      // they get the same singleton floating-mode treatment as an EITHER-class
      // op's output (see below).
      BoolVar mode = solver.addFloatingMode({arg});
      solver.addModeConversionCost(arg, mode);
    }
  }

  // Region-branch successor inputs (loop iter_args, scf.if results, etc.)
  // are collected here so that Step 3b can materialize them once solving is
  // done; see addRegionBranchConstraints for how they're constrained.
  llvm::SetVector<Value> polySuccessorInputs;
  for (RegionBranchOpInterface regionBranchOp : regionBranchOps) {
    addRegionBranchConstraints(solver, regionBranchOp, polySuccessorInputs);
  }

  for (Operation* op : rewriteOrder) {
    auto polyResults = filterPolynomialOps(op->getResults());
    auto polyOperands = filterPolynomialOps(op->getOperands());
    OpFormClass opClass = opFormClass(op);

    // The (polynomial) inputs to ReturnOps get output directly
    // and we choose to not constrain their form in the SAT instance.
    if (opClass == OpFormClass::RETURN) {
      for (Value v : polyOperands) {
        solver.forceDemandEitherForm(v);
      }
    }
    // These ops have coeff-form outputs/inputs
    else if (opClass == OpFormClass::COEFF) {
      if (polyResults.size() == 0) {
        // For ops with no poly outputs, assume the result is needed and force
        // the input to be in coeff form. DCE will remove these ops later
        // if they truly aren't needed
        for (Value v : polyOperands) {
          // since we run DCE at the beginning of this pass, this value *IS*
          // needed in the IR. We can't see that demand because the outputs
          // aren't polynomials, but it is correct/necessary to force the
          // input to coeff form
          solver.forceDemandFixedForm(v, Form::COEFF);
        }
      } else if (polyResults.size() == 1) {
        // For ops with one poly output, the input is needed in coeff form
        // iff the output is needed in coeff form, and if the output is needed
        // in eval form, then it is also needed in coeff form.
        Value y = polyResults[0];
        // Since this op outputs coeff form, the use of eval form implies the
        // use of coeff form
        solver.implyForm(y, Form::EVAL, Form::COEFF);
        // There's a conversion cost if y_e is needed
        solver.addConversionCostForForm(y, Form::EVAL);
        for (Value x : polyOperands) {
          // Use of output in coeff form implies use of input in coeff form
          solver.implyUse(y, x, Form::COEFF);
        }
      } else {
        op->emitOpError(
            "Walk 1: Op has multiple polynomial outputs, but this pass only "
            "handles a single output.");
        signalPassFailure();
        return;
      }
    }
    // Eval poly inputs and outputs; this is really a mirror of the previous
    // case
    else if (opClass == OpFormClass::EVAL) {
      Value y = polyResults[0];
      // Since this op outputs eval form, the use of coeff form implies the
      // use of eval form
      solver.implyForm(y, Form::COEFF, Form::EVAL);
      // There's a conversion cost if y_c is needed
      solver.addConversionCostForForm(y, Form::COEFF);
      for (Value x : polyOperands) {
        // Use of output in eval form implies use of input in eval form
        solver.implyUse(y, x, Form::EVAL);
      }
    }
    // Ops that work in either form, as long as inputs and outputs are all
    // "uni-form"
    else if (opClass == OpFormClass::EITHER) {
      Value y = polyResults[0];
      // Since the value output by this op can be in either form, it gets a
      // 'mode' variable, shared with nothing else (a singleton floating
      // group): the op's inputs must match whatever mode the objective picks
      // for y, and a conversion is charged at y's own site iff y's demand
      // disagrees with that mode.
      BoolVar mode = solver.addFloatingMode({y});
      for (Value x : polyOperands) {
        solver.implyMode(y, x);
      }
      solver.addModeConversionCost(y, mode);
    }
    // Ops that produce polynomials in any form. We can pre-compute these
    // constants in either (or both!) form(s)
    else if (opClass == OpFormClass::CONST) {
      Value y = polyResults[0];
      // Explicitly set the conversion cost of these ops to zero.
      solver.setZeroConversionCost(y);
    } else {
      op->emitOpError(
          "Walk 1: Unexpected op with polynomial inputs/outputs in "
          "polyMulToNTT");
      signalPassFailure();
      return;
    }
  }

  /***************************************************
   ********** Step 2: Solve CP-SAT instance **********
   **************************************************/
  const CPSATSolution soln = solver.solve();
  if (!soln.isValid()) {
    func->emitOpError("Unable to find a feasible solution to CP-SAT instance");
    signalPassFailure();
    return;
  }
  if (!soln.isOptimal()) {
    func->emitWarning(
        "CP-SAT found a FEASIBLE but not provably OPTIMAL solution; the "
        "result may use more NTT/iNTT conversions than necessary");
  }

  /************************************************
   ********** Step 3a: Fix up AST outputs *********
   ***********************************************/
  // In this step, we note the places where the solution says we need
  // a conversion, and add them to the AST. This walk only deals with
  // op *outputs*.

  // A map from input-AST value to AST value in a particular form
  llvm::DenseMap<Value, Value> coeffFormCache;
  llvm::DenseMap<Value, Value> evalFormCache;

  ImplicitLocOpBuilder b(func.getLoc(), rewriter);

  // Given a PolynomialType, output a new Polynomial type with the same ring
  // and the given form
  auto typeToForm = [&](Type ty, Form form) -> Type {
    if (auto p = dyn_cast<PolynomialType>(ty)) {
      return PolynomialType::get(rewriter.getContext(), p.getRing(), form);
    }
    if (auto rt = dyn_cast<RankedTensorType>(ty)) {
      auto elem = dyn_cast<PolynomialType>(rt.getElementType());
      if (!elem) return Type();
      auto newElem =
          PolynomialType::get(elem.getContext(), elem.getRing(), form);
      return RankedTensorType::get(rt.getShape(), newElem, rt.getEncoding());
    }
    func.emitError()
        << "polyMulToNTT:typeToForm expected polynomial-like type, got " << ty;
    return Type();
  };

  // Convert the value v to the given form by adding an NTTOp or INTTOp to the
  // AST.
  auto addConversion = [&](Value& v, Form outputForm) -> Value {
    // Real roots are inserted with the --attach-ntt-roots pass
    if (outputForm == Form::EVAL) {
      ++numNttsInserted;
      return NTTOp::create(b, v, PrimitiveRootAttr()).getOutput();
    } else {
      ++numInttsInserted;
      return INTTOp::create(b, v, PrimitiveRootAttr()).getOutput();
    }
  };

  // First, deal with function arguments. We save the argument types for use in
  // step 5.
  SmallVector<Type> newInputTypes = llvm::to_vector(func.getArgumentTypes());
  b.setInsertionPointToStart(&func.front());
  for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    if (!isPolyValue(arg)) {
      // preserve the type
      newInputTypes[i] = arg.getType();
      continue;
    }

    // The argument's materialized form is whatever the solver's (freely
    // floating) mode picked for it -- see the addFloatingMode call in Step 1.
    // If the other form is *also* needed, we obtain it via a conversion right
    // here at the argument's definition site.
    Form f = soln.getMode(arg);
    Type newTy = typeToForm(arg.getType(), f);
    if (!newTy) {
      signalPassFailure();
      return;
    }
    newInputTypes[i] = newTy;
    // set the type of the argument SSA value
    arg.setType(newTy);
    if (f == Form::COEFF) {
      coeffFormCache[arg] = arg;
      if (soln.needsForm(arg, Form::EVAL)) {
        evalFormCache[arg] = addConversion(arg, Form::EVAL);
      }
    } else {
      evalFormCache[arg] = arg;
      if (soln.needsForm(arg, Form::COEFF)) {
        coeffFormCache[arg] = addConversion(arg, Form::COEFF);
      }
    }
  }

  // Now walk the poly-op tree and add conversions on outputs where needed
  // There are a lot of sanity checks here that could be removed
  for (Operation* op : rewriteOrder) {
    auto polyResults = filterPolynomialOps(op->getResults());
    if (polyResults.size() == 0) {
      // no polynomial outputs, so nothing to do
      // This includes func::ReturnOp, ToTensorOp, LeadingTermOp, EvalOp
      continue;
    }
    OpFormClass opClass = opFormClass(op);

    b.setInsertionPointAfter(op);
    // Coeff poly outputs
    if (opClass == OpFormClass::COEFF) {
      Value v = polyResults[0];
      if (!soln.needsForm(v, Form::COEFF)) {
        // Sanity check: this should be forced in the solution
        op->emitOpError(
            "Walk 2: CP-SAT soln does not require coeff-form output for "
            "coeff-form op");
        signalPassFailure();
        return;
      }
      if (soln.needsForm(v, Form::EVAL) != soln.needsConversion(v)) {
        // Sanity check: Since this op outputs coeff-form outputs, eval form is
        // needed iff a conversion is needed
        op->emitOpError(
            "Walk 2: CP-SAT soln mandates eval form output or conversion for "
            "coeff-form output, but not both");
        signalPassFailure();
        return;
      }
      coeffFormCache[v] = v;
      if (soln.needsForm(v, Form::EVAL)) {
        evalFormCache[v] = addConversion(v, Form::EVAL);
      }
    }
    // Eval poly outputs
    else if (opClass == OpFormClass::EVAL) {
      Value v = polyResults[0];
      if (soln.needsForm(v, Form::COEFF) && !soln.needsForm(v, Form::EVAL)) {
        // Sanity check: this should be forced in the solution
        op->emitOpError(
            "Walk 2: CP-SAT soln does not require eval-form output for "
            "eval-form op");
        signalPassFailure();
        return;
      }
      if (soln.needsForm(v, Form::COEFF) != soln.needsConversion(v)) {
        // Sanity check: Since this op outputs eval-form outputs, coeff form is
        // needed iff a conversion is needed
        op->emitOpError(
            "Walk 2: CP-SAT soln mandates coeff form output or conversion for "
            "eval-form output, but not both");
        signalPassFailure();
        return;
      }
      // The result type is coeff form; update it to be eval form
      Type newTy = typeToForm(v.getType(), Form::EVAL);
      if (!newTy) {
        signalPassFailure();
        return;
      }
      op->getResult(0).setType(newTy);
      evalFormCache[v] = v;
      if (soln.needsForm(v, Form::COEFF)) {
        coeffFormCache[v] = addConversion(v, Form::COEFF);
      }
    }
    // Ops that work in either form, as long as inputs and outputs are all
    // "uni-form"
    else if (opClass == OpFormClass::EITHER) {
      Value v = polyResults[0];
      bool modeMismatch =
          (soln.getMode(v) == Form::EVAL && soln.needsForm(v, Form::COEFF)) ||
          (soln.getMode(v) == Form::COEFF && soln.needsForm(v, Form::EVAL));
      if (soln.needsConversion(v) != modeMismatch) {
        // Sanity check: This is explicitly encoded into the CP-SAT instance, so
        // it should always be satisfied
        op->emitOpError(
            "Walk 2: CP-SAT soln mandates a conversion iff v's own demand "
            "disagrees with its mode");
        signalPassFailure();
        return;
      }
      if (!soln.needsConversion(v)) {
        if (soln.needsForm(v, Form::COEFF) && soln.getMode(v) != Form::COEFF) {
          // Sanity check: since no conversion is needed, v_coeff must be
          // needed, and we should operate this op in coeff mode
          op->emitOpError(
              "Walk 2: Only coeff output is needed for a flexibile op, but "
              "mode does not match the output form");
          signalPassFailure();
          return;
        }
        if (soln.needsForm(v, Form::EVAL) && soln.getMode(v) != Form::EVAL) {
          // Sanity check: since no conversion is needed, v_coeff must be
          // needed, and we should operate this op in coeff mode
          op->emitOpError(
              "Walk 2: Only eval output is needed for a flexibile op, but mode "
              "does not match the output form");
          signalPassFailure();
          return;
        }
      }

      if (soln.getMode(v) == Form::COEFF) {
        coeffFormCache[v] = v;
        // The easy case: just convert the output if needed
        if (soln.needsForm(v, Form::EVAL)) {
          evalFormCache[v] = addConversion(v, Form::EVAL);
        }
      } else {
        Type newTy = typeToForm(v.getType(), Form::EVAL);
        if (!newTy) {
          signalPassFailure();
          return;
        }
        op->getResult(0).setType(newTy);
        evalFormCache[v] = v;
        if (soln.needsForm(v, Form::COEFF)) {
          coeffFormCache[v] = addConversion(v, Form::COEFF);
        }
      }
    }
    // Ops that produce polynomials in any form
    else if (opClass == OpFormClass::CONST) {
      Value v = polyResults[0];
      if (soln.needsConversion(v)) {
        // Sanity check: we never require explicit conversions for constants;
        // conversions for constants are computed at compile-time via
        // constant-folding
        op->emitOpError(
            "Walk 2: CP-SAT soln requires conversion for constant; this should "
            "be prohibited");
        signalPassFailure();
        return;
      }

      Type coeffTy = typeToForm(v.getType(), Form::COEFF);
      Type evalTy = typeToForm(v.getType(), Form::EVAL);
      if (!coeffTy || !evalTy) {
        signalPassFailure();
        return;
      }

      auto repairConstantValueAttr = [&](ConstantOp constantOp, Type newTy) {
        Attribute value = constantOp.getValue();
        llvm::TypeSwitch<Attribute>(value)
            .Case<TypedIntPolynomialAttr>([&](auto intAttr) {
              constantOp->setAttr("value", TypedIntPolynomialAttr::get(
                                               newTy, intAttr.getValue()));
            })
            .Case<TypedFloatPolynomialAttr>([&](auto floatAttr) {
              constantOp->setAttr("value", TypedFloatPolynomialAttr::get(
                                               newTy, floatAttr.getValue()));
            })
            .Case<RNSPolynomialAttr>([&](auto rnsAttr) {
              auto newPolyTy = cast<PolynomialType>(newTy);
              constantOp->setAttr(
                  "value", RNSPolynomialAttr::get(constantOp.getContext(),
                                                  rnsAttr.getCoefficients(),
                                                  newTy, newPolyTy.getForm()));
            })
            .Default([](Attribute) {});
      };

      if (soln.needsForm(v, Form::COEFF)) {
        op->getResult(0).setType(coeffTy);
        if (auto constantOp = dyn_cast<ConstantOp>(op)) {
          repairConstantValueAttr(constantOp, coeffTy);
        }
        coeffFormCache[v] = v;
      } else {
        op->getResult(0).setType(evalTy);
        if (auto constantOp = dyn_cast<ConstantOp>(op)) {
          repairConstantValueAttr(constantOp, evalTy);
        }
        evalFormCache[v] = v;
      }

      // If we get here, we already materialized COEFF form, so we just need
      // EVAL form
      if (soln.needsForm(v, Form::COEFF) && soln.needsForm(v, Form::EVAL)) {
        Operation* evalOp = b.clone(*op);
        evalOp->getResult(0).setType(evalTy);
        if (auto constantOp = dyn_cast<ConstantOp>(evalOp)) {
          repairConstantValueAttr(constantOp, evalTy);
        }
        evalFormCache[v] = evalOp->getResult(0);
      }
    } else {
      op->emitOpError(
          "Walk 2: Unexpected op with polynomial inputs/outputs in "
          "polyMulToNTT");
      signalPassFailure();
      return;
    }
  }

  /****************************************************************
   ***** Step 3b: Materialize region-branch successor inputs ******
   *****************************************************************/
  // Block arguments and region-branch results are materialized exactly like
  // function arguments (see the loop over func.getArguments() above): we
  // fix the native form the solver's shared mode picked for this successor
  // input's group into the IR, and, if the other form is also required by
  // some use, insert a single conversion right where the value comes into
  // existence: at the start of the owning block for a block argument, or
  // right after the op for a value returned to the parent. We remember which
  // form each successor input was fixed to so Step 4b can rewrite the
  // forwarding operands that feed it to match.
  llvm::DenseMap<Value, Form> successorNativeForm;
  for (Value target : polySuccessorInputs) {
    Form f = soln.getMode(target);
    successorNativeForm[target] = f;

    Type newTy = typeToForm(target.getType(), f);
    if (!newTy) {
      signalPassFailure();
      return;
    }
    target.setType(newTy);

    if (auto blockArg = dyn_cast<BlockArgument>(target)) {
      b.setInsertionPointToStart(blockArg.getOwner());
    } else {
      b.setInsertionPointAfter(target.getDefiningOp());
    }

    if (f == Form::COEFF) {
      coeffFormCache[target] = target;
      if (soln.needsForm(target, Form::EVAL)) {
        evalFormCache[target] = addConversion(target, Form::EVAL);
      }
    } else {
      evalFormCache[target] = target;
      if (soln.needsForm(target, Form::COEFF)) {
        coeffFormCache[target] = addConversion(target, Form::COEFF);
      }
    }
  }

  /************************************************
   *********** Step 4a: Fix up AST inputs *********
   ***********************************************/
  // We have pre-populated the cache, so all inputs that are required
  // have been created and are in the AST. In this step, we point op
  // inputs to the correct value/form.

  // Given an input-AST value and a target form, output the corresponding
  // AST value with that form
  auto formToValue = [&](const Value& v, Form form) -> Value {
    if (form == Form::COEFF) {
      return coeffFormCache.at(v);
    } else {
      return evalFormCache.at(v);
    }
  };

  // pre-computation for step 5: we populate the new set of result types for the
  // function based on the inputs to ReturnOp(s).
  SmallVector<Type> newResultTypes = llvm::to_vector(func.getResultTypes());

  // Walk the AST
  for (Operation* op : rewriteOrder) {
    llvm::SmallVector<OpOperand*> polyOperands;
    for (OpOperand& arg : op->getOpOperands()) {
      if (isPolyValue(arg.get())) {
        polyOperands.push_back(&arg);
      }
    }

    auto polyResults = filterPolynomialOps(op->getResults());

    if (polyOperands.size() == 0) {
      // no polynomial inputs, so nothing to do
      // This includes MonicMonomialMulOp, FromTensorOp, MonomialOp, ConstantOp
      continue;
    }

    OpFormClass opClass = opFormClass(op);
    b.setInsertionPoint(op);

    if (opClass == OpFormClass::RETURN) {
      for (OpOperand* arg : polyOperands) {
        Value v = arg->get();
        Form form = Form::EVAL;
        // Like the argument "problem" noted above, we may have to make an
        // choice here. If a return value is available in both forms,
        // we (arbitrarily) prefer coeff form. As with function inputs,
        // this is not necessarily optimal (from the caller's perspective).
        if (soln.needsForm(v, Form::COEFF)) {
          form = Form::COEFF;
        } else if (!soln.needsForm(v, Form::EVAL)) {
          op->emitOpError(
              "Walk 3: Input to return has neither form materialized");
          signalPassFailure();
          return;
        }
        arg->set(formToValue(v, form));
        newResultTypes[arg->getOperandNumber()] =
            formToValue(v, form).getType();
      }
    } else if (opClass == OpFormClass::COEFF) {
      // Ops that always take COEFF inputs
      for (OpOperand* arg : polyOperands) {
        arg->set(formToValue(arg->get(), Form::COEFF));
      }
    }
    // Ops that always take EVAL inputs
    else if (opClass == OpFormClass::EVAL) {
      for (OpOperand* arg : polyOperands) {
        arg->set(formToValue(arg->get(), Form::EVAL));
      }
    }
    // Ops that work in either form, as long as inputs and outputs are all
    // "uni-form"
    else if (opClass == OpFormClass::EITHER) {
      Value v = polyResults[0];
      Form form = soln.getMode(v);
      for (OpOperand* arg : polyOperands) {
        arg->set(formToValue(arg->get(), form));
      }
    } else {
      op->emitOpError(
          "Walk 3: Unexpected op with polynomial inputs/outputs in "
          "polyMulToNTT");
      signalPassFailure();
      return;
    }
  }

  /**********************************************************
   ***** Step 4b: Rewrite region-branch forwarding edges *****
   **********************************************************/
  // Every successor input now has a resolved, materialized native form
  // (Step 3b). Point each operand that forwards into one -- the region
  // branch op's own entry operands, and every operand yielded/forwarded
  // inside its regions -- at the cached value in that form.
  for (RegionBranchOpInterface regionBranchOp : regionBranchOps) {
    RegionBranchSuccessorMapping operandToInputs;
    regionBranchOp.getSuccessorOperandInputMapping(operandToInputs);
    for (auto& [operand, inputs] : operandToInputs) {
      if (!isPolyValue(operand->get())) continue;
      // All of this operand's targets share one mode variable (set up in
      // addRegionBranchConstraints via addFloatingMode), so any one of them
      // tells us the form this operand must be rewritten to.
      Form form = successorNativeForm.at(inputs.front());
      operand->set(formToValue(operand->get(), form));
    }
  }

  /************************************************
   ********* Step 5: Fix function signature *******
   ***********************************************/
  // We have to fix the function signature itself. We saved the types
  // of the arguments in step 3 and the types of the results in step 4,
  // so we use them here.

  // Consistency check on return values
  func.walk([&](func::ReturnOp ret) {
    for (auto [i, output] : llvm::enumerate(ret->getOperands())) {
      if (output.getType() != newResultTypes[i]) {
        ret->emitOpError("Function return types are inconsistent");
        signalPassFailure();
        return;
      }
    }
  });

  rewriter.modifyOpInPlace(func, [&] {
    func.setFunctionType(
        rewriter.getFunctionType(newInputTypes, newResultTypes));
  });
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
