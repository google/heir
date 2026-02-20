#include "lib/Dialect/Polynomial/Transforms/PolyMulToNTT.h"

#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/Transforms/NTTSolver.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_POLYMULTONTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

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

// If the type is a PolynomialType, return it. If it's a container of
// polynomials, extract the polynomial element type
static PolynomialType getPolyElementType(Type t) {
  if (auto p = dyn_cast<PolynomialType>(t)) return p;
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    return dyn_cast<PolynomialType>(rt.getElementType());
  }
  return {};
}

void PolyMulToNTT::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* context = &getContext();

  NTTSolver solver;

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
  // program) and find an optimal solution that miniimzes the number of
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

  // Steps 1, 3, and 4 above involve walking the AST. Since we're going to be
  // doing multiple walks and adding some nodes on the way, we first memoize
  // the AST (so that we're not walking and mutating at the same time) and
  // prune it to remove ops that don't involve polynomials. This doesn't remove
  // ops from the AST, it just means that we don't walk over them later.
  llvm::SmallVector<Operation*> rewriteOrder;
  WalkResult wr =
      func.walk([&](Operation* op) -> WalkResult {
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

  for (Operation* op : rewriteOrder) {
    auto polyResults = filterPolynomialOps(op->getResults());
    auto polyOperands = filterPolynomialOps(op->getOperands());

    // The (polynomial) inputs to ReturnOps get output directly
    // and we choose to not constrain their form in the SAT instance.
    if (isa<func::ReturnOp>(op)) {
      for (Value v : polyOperands) {
        solver.allowEitherForm(v);
      }
    }
    // These ops have coeff-form inputs and no (poly) outputs
    // We therefore require their operands to be available in coeff form.
    else if (isa<ToTensorOp, LeadingTermOp, EvalOp>(op)) {
      for (Value v : polyOperands) {
        solver.fixForm(v, Form::COEFF);
      }
    }
    // These ops have coeff-form outputs (and possibly coeff-form inputs)
    else if (isa<ConvertBasisOp, MonicMonomialMulOp, FromTensorOp>(op)) {
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
    }
    // Eval poly inputs and outputs; this is really a mirror of the previous
    // case
    else if (isa<MulOp>(op)) {
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
    else if (isa<AddOp, SubOp, MulScalarOp, ModSwitchOp, ExtractSliceOp,
                 tensor::ExtractSliceOp, tensor::ExtractOp,
                 tensor::FromElementsOp>(op)) {
      Value y = polyResults[0];
      // Since the value output by this op can be in either form, it gets a
      // 'mode' variable. In short, if y_c is needed and y_e is not, we run the
      // op in coeff mode, and vice versa.
      solver.addOpMode(y);
      for (Value x : polyOperands) {
        // if y_mode = 0 and output (in either form) is needed, the inputs in
        // coeff form are required if y_mode = 1 and output (in either form) is
        // needed, the inputs in eval form are required
        solver.implyMode(y, x);
      }
      // The only time there's a conversion cost is if both forms are needed. If
      // only one form is needed, the op runs in that mode.
      solver.addConversionCostIfBothForms(y);
    }
    // Ops that produce polynomials in any form. We can pre-compute these
    // constants in either (or both!) form(s)
    else if (isa<MonomialOp, ConstantOp>(op)) {
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
    func->emitOpError("Unable to find solution to CP-SAT instance");
    signalPassFailure();
    return;
  }

  /************************************************
   ********** Step 3: Fix up AST outputs **********
   ***********************************************/
  // In this step, we note the places where the solution says we need
  // a conversion, and add them to the AST. This walk only deals with
  // op *outputs*.

  // A map from input-AST value to AST value in a particular form
  llvm::DenseMap<Value, Value> coeffFormCache;
  llvm::DenseMap<Value, Value> evalFormCache;

  IRRewriter rewriter(context);
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

  // Convert the value v to the given form. This adds an op to the AST.
  // Do NOT convert if the input is a tensor.
  auto addConversion = [&](Value& v, Form outputForm) -> Value {
    // TODO need to insert PrimitiveRootAttrs here
    if (outputForm == Form::EVAL) {
      return NTTOp::create(b, v, PrimitiveRootAttr()).getOutput();
    } else {
      return INTTOp::create(b, v, PrimitiveRootAttr()).getOutput();
    }
  };

  // First, deal with function arguments. We save the argument types
  // for use in step 5.
  SmallVector<Type> newInputTypes = llvm::to_vector(func.getArgumentTypes());
  if (!func.isDeclaration()) {
    b.setInsertionPointToStart(&func.front());
  }
  for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    if (!isPolyValue(arg)) {
      // preserve the type
      newInputTypes[i] = arg.getType();
      continue;
    }

    // If the function naturally needs both forms of an input, we have to
    // arbitrarily pick one to be in the signature. Here, we say "if the
    // coefficient form of this input is needed, put that in the signature".
    // If the eval form is *also* needed, we'll obtain it via an NTT of the
    // input.
    Form f = soln.needsForm(arg, Form::COEFF) ? Form::COEFF : Form::EVAL;
    Type newTy = typeToForm(arg.getType(), f);
    if (!newTy) {
      signalPassFailure();
      return;
    }
    newInputTypes[i] = newTy;
    if (!func.isDeclaration()) {
      // set the type of the argument SSA value
      arg.setType(newTy);
    }
    if (f == Form::COEFF) {
      // The coeff-form of this value is the argument itself
      coeffFormCache[arg] = arg;
      // if the solution also requires the eval form, add an NTT
      // and cache the result
      if (soln.needsForm(arg, Form::EVAL)) {
        evalFormCache[arg] = addConversion(arg, Form::EVAL);
      }
    } else {
      evalFormCache[arg] = arg;
      // because of our arbitrary choice above, we know that coeff form is NOT
      // required
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

    b.setInsertionPointAfter(op);
    // Coeff poly outputs
    if (isa<ConvertBasisOp, MonicMonomialMulOp, FromTensorOp>(op)) {
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
    else if (isa<MulOp>(op)) {
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
    else if (isa<AddOp, SubOp, MulScalarOp, ModSwitchOp, ExtractSliceOp,
                 tensor::ExtractOp, tensor::ExtractSliceOp,
                 tensor::FromElementsOp>(op)) {
      Value v = polyResults[0];
      if (soln.needsConversion(v) !=
          (soln.needsForm(v, Form::COEFF) && soln.needsForm(v, Form::EVAL))) {
        // Sanity check: This is explicitly encoded into the CP-SAT instance, so
        // it should always be satisfied
        op->emitOpError(
            "Walk 2: CP-SAT soln mandates coeff form output or conversion for "
            "eval-form output, but not both");
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
    else if (isa<MonomialOp, ConstantOp>(op)) {
      Value v = polyResults[0];
      if (soln.needsConversion(v)) {
        // Sanity check: we never require explicit conversions for constants;
        // they should be precomputed
        op->emitOpError(
            "Walk 2: CP-SAT soln requires conversion for constant; this should "
            "be prohibited");
        signalPassFailure();
        return;
      }
      // Keep constants/monomials in coeff form and materialize eval via NTT
      // when needed. This avoids type-inference mismatches on ConstantOp.
      coeffFormCache[v] = v;
      if (soln.needsForm(v, Form::EVAL)) {
        evalFormCache[v] = addConversion(v, Form::EVAL);
      }
    } else {
      op->emitOpError(
          "Walk 2: Unexpected op with polynomial inputs/outputs in "
          "polyMulToNTT");
      signalPassFailure();
      return;
    }
  }

  /************************************************
   *********** Step 4: Fix up AST inputs **********
   ***********************************************/
  // We have pre-populated the cache, so all inputs that are required
  // have been created and are in the AST. In this step, we point op
  // inputs to the correct value/form.

  // Given an input-AST value and a target form, output the corresponding
  // AST value with that form
  auto formToValue = [&](Value& v, Form form) -> Value {
    if (form == Form::COEFF) {
      return coeffFormCache[v];
    } else {
      return evalFormCache[v];
    }
  };

  // pre-computation for step 5: we populate the new set of result types for the
  // function based on the inputs to ReturnOp(s).
  SmallVector<Type> newResultTypes = llvm::to_vector(func.getResultTypes());

  // Walk the AST
  for (Operation* op : rewriteOrder) {
    auto polyOperands = filterPolynomialOps(op->getOperands());
    auto polyResults = filterPolynomialOps(op->getResults());

    if (polyOperands.size() == 0) {
      // no polynomial inputs, so nothing to do
      // This includes MonicMonomialMulOp, FromTensorOp, MonomialOp, ConstantOp
      continue;
    }

    b.setInsertionPoint(op);

    if (isa<func::ReturnOp>(op)) {
      // We have to iterate over *all* operands here, because some might not be
      // polynomials. If you only iterate over polyResults, the indexing will
      // be wrong.
      for (auto [i, v] : llvm::enumerate(op->getOperands())) {
        if (!isPolyValue(v)) {
          // Not a polynomial, so don't change the type or value of this operand
          continue;
        }
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
        op->setOperand(i, formToValue(v, form));
        newResultTypes[i] = formToValue(v, form).getType();
      }
    } else if (isa<ConvertBasisOp, MonicMonomialMulOp, ToTensorOp,
                   LeadingTermOp, EvalOp>(op)) {
      // Ops that always take COEFF inputs
      for (auto [i, v] : llvm::enumerate(op->getOperands())) {
        if (!getPolyElementType(v.getType())) {
          continue;
        }
        op->setOperand(i, formToValue(v, Form::COEFF));
      }
    }
    // Eval poly inputs and outputs
    else if (isa<MulOp>(op)) {
      // Ops that always take EVAL inputs
      for (auto [i, v] : llvm::enumerate(op->getOperands())) {
        if (!getPolyElementType(v.getType())) {
          continue;
        }
        op->setOperand(i, formToValue(v, Form::EVAL));
      }
    }
    // Ops that work in either form, as long as inputs and outputs are all
    // "uni-form"
    else if (isa<AddOp, SubOp, MulScalarOp, ModSwitchOp, ExtractSliceOp,
                 tensor::ExtractOp, tensor::ExtractSliceOp,
                 tensor::FromElementsOp>(op)) {
      Value v = polyResults[0];
      Form form = soln.getMode(v);
      for (auto [i, v] : llvm::enumerate(op->getOperands())) {
        if (!getPolyElementType(v.getType())) {
          continue;
        }
        op->setOperand(i, formToValue(v, form));
      }
    } else {
      op->emitOpError(
          "Walk 3: Unexpected op with polynomial inputs/outputs in "
          "polyMulToNTT");
      signalPassFailure();
      return;
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
