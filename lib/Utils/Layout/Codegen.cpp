#include "lib/Utils/Layout/Codegen.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/APInt.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DynamicAPInt.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"             // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project

// ISL
#include "include/isl/ast.h"             // from @isl
#include "include/isl/ast_build.h"       // from @isl
#include "include/isl/ast_type.h"        // from @isl
#include "include/isl/constraint.h"      // from @isl
#include "include/isl/ctx.h"             // from @isl
#include "include/isl/id.h"              // from @isl
#include "include/isl/id_type.h"         // from @isl
#include "include/isl/map.h"             // from @isl
#include "include/isl/map_type.h"        // from @isl
#include "include/isl/set.h"             // from @isl
#include "include/isl/space.h"           // from @isl
#include "include/isl/space_type.h"      // from @isl
#include "include/isl/union_map.h"       // from @isl
#include "include/isl/union_map_type.h"  // from @isl
#include "include/isl/val.h"             // from @isl
#include "include/isl/val_type.h"        // from @isl

namespace mlir {
namespace heir {

namespace {

static std::map<isl_ast_expr_op_type, std::string> islBinaryOpToMlir = {
    {isl_ast_op_add, "arith.addi"},
    {isl_ast_op_sub, "arith.subi"},
    {isl_ast_op_mul, "arith.muli"},
    {isl_ast_op_div, "arith.divsi"},
    {isl_ast_op_pdiv_q, "arith.floordivsi"},
    {isl_ast_op_max, "arith.maxsi"},
    {isl_ast_op_min, "arith.minsi"},
    {isl_ast_op_and, "arith.andi"},
    {isl_ast_op_or, "arith.ori"},
    {isl_ast_op_fdiv_q, "arith.floordivsi"},
};

static std::map<isl_ast_expr_op_type, arith::CmpIPredicate> islCmpToMlirAttr = {
    {isl_ast_op_eq, arith::CmpIPredicate::eq},
    {isl_ast_op_lt, arith::CmpIPredicate::slt},
    {isl_ast_op_le, arith::CmpIPredicate::sle},
    {isl_ast_op_gt, arith::CmpIPredicate::sgt},
    {isl_ast_op_ge, arith::CmpIPredicate::sge},
};

}  // namespace

// Converts a basic map (representing proper domain/range vars as in/out vars)
// to a schedule that maps all domain+range vars to range vars.
//
// In particular, this tells the ISL codegen that we want to iterate over just
// the range variables. This is specific to FHE because we know the packing must
// be a partial function from range (ct, slot) to domain (data indices).
//
// E.g., for an input basic map like
//
//   {
//      [row,col] -> [ct,slot] :
//      0 <= row,ct < 4
//      and 0 <= col < 8
//      and 0 <= slot < 32
//      and ((-row + slot) % 4) = 0 and (-col + ct + slot) % 8 = 0
//   }
//
// The output schedule would have the same constraints, but the map would start
// with
//
//   S[row,col,ct,slot] -> [ct,slot]
//
// If domainIndicesToSchedule is provided, those domain indices will be added to
// the range of the schedule map, resulting in outer loops for those indices.
static __isl_give isl_union_map* createSchedule(
    __isl_keep isl_map* map, isl_ctx* ctx,
    ArrayRef<int> domainIndicesToSchedule) {
  unsigned int numIn = isl_map_dim(map, isl_dim_in);
  unsigned int numOut = isl_map_dim(map, isl_dim_out);

  // The schedule is a map from the original domain + original range to the
  // schedule space (which includes the original range and any requested domain
  // indices).
  isl_map* scheduleMap = isl_map_copy(map);

  // Add the range variables as additional domain variables to the end of
  // the domain.
  scheduleMap = isl_map_insert_dims(scheduleMap, isl_dim_in, numIn, numOut);

  // Add constraints to equate the new domain dimensions with the original range
  // dimensions.
  for (unsigned i = 0; i < numOut; ++i) {
    isl_constraint* c = isl_constraint_alloc_equality(
        isl_local_space_from_space(isl_map_get_space(scheduleMap)));
    // range dimension i
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
    // new domain dimension numIn + i
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, numIn + i, -1);
    scheduleMap = isl_map_add_constraint(scheduleMap, c);
  }

  // Now adjust the range of the scheduleMap to include domainIndicesToSchedule.
  // The original range dims are at the end. We insert the new ones at the
  // front.
  scheduleMap = isl_map_insert_dims(scheduleMap, isl_dim_out, 0,
                                    domainIndicesToSchedule.size());

  for (unsigned i = 0; i < domainIndicesToSchedule.size(); ++i) {
    int domainIdx = domainIndicesToSchedule[i];
    isl_constraint* c = isl_constraint_alloc_equality(
        isl_local_space_from_space(isl_map_get_space(scheduleMap)));
    // new range dimension i
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
    // original domain dimension domainIdx
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, domainIdx, -1);
    scheduleMap = isl_map_add_constraint(scheduleMap, c);
  }

  // Set the domain tuple name to "S". This will be used in codegen for the
  // statement to be executed, e.g., S(a, b, c, d)
  scheduleMap = isl_map_set_tuple_name(scheduleMap, isl_dim_in, "S");
  return isl_union_map_from_map(scheduleMap);
}

// Builds the AST that executes `schedule`.
static __isl_give isl_ast_node* buildAst(__isl_take isl_union_map* schedule,
                                         isl_ctx* ctx) {
  // Context and options are intentionally empty. We don't need any of these
  // features, though I admit I have not looked into what they can provide.
  isl_set* context = isl_set_universe(isl_space_params_alloc(ctx, 0));
  isl_union_map* options = isl_union_map_empty(isl_space_params_alloc(ctx, 0));

  // Build the AST with options tuned for performance.
  isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
  isl_options_set_ast_build_detect_min_max(ctx, 0);
  isl_options_set_ast_build_prefer_pdiv(ctx, 1);
  isl_options_set_ast_build_separation_bounds(
      ctx, ISL_AST_BUILD_SEPARATION_BOUNDS_IMPLICIT);
  isl_options_set_ast_build_group_coscheduled(ctx, 0);
  isl_options_set_ast_build_scale_strides(ctx, 1);
  isl_options_set_ast_build_exploit_nested_bounds(ctx, 1);

  isl_ast_build* build = isl_ast_build_from_context(context);
  build = isl_ast_build_set_options(build, options);
  isl_ast_node* tree = isl_ast_build_node_from_schedule_map(build, schedule);
  isl_ast_build_free(build);

  return tree;
}

__isl_give isl_ast_node* constructAst(const presburger::IntegerRelation& rel,
                                      isl_ctx* ctx,
                                      ArrayRef<int> domainIndicesToSchedule) {
  isl_basic_map* bmap = convertRelationToBasicMap(rel, ctx);

  // Simplify the underlying map to speed up AST generation.
  isl_map* map = isl_map_from_basic_map(bmap);
  map = isl_map_compute_divs(map);
  map = isl_map_detect_equalities(map);
  map = isl_map_remove_redundancies(map);
  map = isl_map_coalesce(map);

  isl_union_map* schedule = createSchedule(map, ctx, domainIndicesToSchedule);
  isl_map_free(map);

  schedule = isl_union_map_detect_equalities(schedule);
  schedule = isl_union_map_coalesce(schedule);

  return buildAst(schedule, ctx);
}

namespace {

using llvm::DynamicAPInt;

// An isl AST expression, with each loop iterator replaced by its index into
// the vector of iterator values, so it can be evaluated without isl.
struct ScanExpr {
  enum Kind { kInt, kIterator, kOp } kind = kInt;
  DynamicAPInt value;
  unsigned iterator = 0;
  isl_ast_expr_op_type op = isl_ast_expr_op_error;
  std::vector<ScanExpr> args;
};

// An isl AST node in the same form. A for node's body and an if node's
// branches are its children.
struct ScanNode {
  enum Kind { kFor, kIf, kBlock, kUser } kind = kBlock;
  unsigned iterator = 0;
  ScanExpr init, cond, inc;
  std::vector<ScanNode> children;
  std::vector<ScanExpr> args;
};

// Converts an isl integer exactly. isl's AST can contain constants outside
// the int64_t range even when the relation's coefficients and points fit in
// it.
DynamicAPInt toDynamicAPInt(__isl_keep isl_val* val) {
  if (isl_val_cmp_si(val, std::numeric_limits<int64_t>::max()) <= 0 &&
      isl_val_cmp_si(val, std::numeric_limits<int64_t>::min()) >= 0)
    return DynamicAPInt(isl_val_get_num_si(val));
  char* str = isl_val_to_str(val);
  StringRef digits(str);
  bool negative = digits.consume_front("-");
  // One bit more than the magnitude needs keeps the value positive.
  unsigned bits = llvm::APInt::getBitsNeeded(digits, 10) + 1;
  llvm::APInt magnitude(bits, digits, 10);
  free(str);
  return DynamicAPInt(negative ? -magnitude : magnitude);
}

class AstScanner {
 public:
  ScanNode buildNode(__isl_keep isl_ast_node* node) {
    ScanNode result;
    switch (isl_ast_node_get_type(node)) {
      case isl_ast_node_for: {
        result.kind = ScanNode::kFor;
        isl_ast_expr* iter = isl_ast_node_for_get_iterator(node);
        isl_id* id = isl_ast_expr_get_id(iter);
        result.iterator = getIterator(isl_id_get_name(id));
        isl_id_free(id);
        isl_ast_expr_free(iter);
        result.init = takeExpr(isl_ast_node_for_get_init(node));
        result.cond = takeExpr(isl_ast_node_for_get_cond(node));
        result.inc = takeExpr(isl_ast_node_for_get_inc(node));
        isl_ast_node* body = isl_ast_node_for_get_body(node);
        result.children.push_back(buildNode(body));
        isl_ast_node_free(body);
        return result;
      }
      case isl_ast_node_if: {
        result.kind = ScanNode::kIf;
        result.cond = takeExpr(isl_ast_node_if_get_cond(node));
        isl_ast_node* thenNode = isl_ast_node_if_get_then(node);
        result.children.push_back(buildNode(thenNode));
        isl_ast_node_free(thenNode);
        if (isl_ast_node_if_has_else(node) == isl_bool_true) {
          isl_ast_node* elseNode = isl_ast_node_if_get_else(node);
          result.children.push_back(buildNode(elseNode));
          isl_ast_node_free(elseNode);
        }
        return result;
      }
      case isl_ast_node_block: {
        result.kind = ScanNode::kBlock;
        isl_ast_node_list* list = isl_ast_node_block_get_children(node);
        for (int i = 0; i < isl_ast_node_list_n_ast_node(list); ++i) {
          isl_ast_node* child = isl_ast_node_list_get_ast_node(list, i);
          result.children.push_back(buildNode(child));
          isl_ast_node_free(child);
        }
        isl_ast_node_list_free(list);
        return result;
      }
      case isl_ast_node_mark: {
        isl_ast_node* child = isl_ast_node_mark_get_node(node);
        result = buildNode(child);
        isl_ast_node_free(child);
        return result;
      }
      case isl_ast_node_user: {
        // The statement is a call S(coordinates...); argument 0 is S.
        result.kind = ScanNode::kUser;
        isl_ast_expr* call = isl_ast_node_user_get_expr(node);
        for (int i = 1; i < isl_ast_expr_get_op_n_arg(call); ++i)
          result.args.push_back(takeExpr(isl_ast_expr_get_op_arg(call, i)));
        isl_ast_expr_free(call);
        return result;
      }
      default:
        llvm::report_fatal_error("forEachPoint: unexpected isl AST node");
    }
  }

  unsigned numIterators() const { return iterators.size(); }

 private:
  unsigned getIterator(const std::string& name) {
    auto [it, inserted] = iterators.try_emplace(name, iterators.size());
    return it->second;
  }

  ScanExpr takeExpr(__isl_take isl_ast_expr* expr) {
    ScanExpr result;
    switch (isl_ast_expr_get_type(expr)) {
      case isl_ast_expr_int: {
        isl_val* val = isl_ast_expr_get_val(expr);
        result.kind = ScanExpr::kInt;
        result.value = toDynamicAPInt(val);
        isl_val_free(val);
        break;
      }
      case isl_ast_expr_id: {
        isl_id* id = isl_ast_expr_get_id(expr);
        result.kind = ScanExpr::kIterator;
        result.iterator = getIterator(isl_id_get_name(id));
        isl_id_free(id);
        break;
      }
      case isl_ast_expr_op: {
        result.kind = ScanExpr::kOp;
        result.op = isl_ast_expr_get_op_type(expr);
        for (int i = 0; i < isl_ast_expr_get_op_n_arg(expr); ++i)
          result.args.push_back(takeExpr(isl_ast_expr_get_op_arg(expr, i)));
        break;
      }
      default:
        llvm::report_fatal_error("forEachPoint: unexpected isl AST expression");
    }
    isl_ast_expr_free(expr);
    return result;
  }

  std::map<std::string, unsigned> iterators;
};

// Evaluates `expr` exactly: intermediate values of an isl AST may leave the
// int64_t range even when every coordinate fits in it.
DynamicAPInt evaluate(const ScanExpr& expr, ArrayRef<DynamicAPInt> iterators) {
  switch (expr.kind) {
    case ScanExpr::kInt:
      return expr.value;
    case ScanExpr::kIterator:
      return iterators[expr.iterator];
    case ScanExpr::kOp:
      break;
  }
  auto arg = [&](unsigned i) { return evaluate(expr.args[i], iterators); };
  auto boolean = [](bool b) { return DynamicAPInt(b); };
  switch (expr.op) {
    case isl_ast_expr_op_and:
    case isl_ast_expr_op_and_then:
      return boolean(arg(0) != 0 && arg(1) != 0);
    case isl_ast_expr_op_or:
    case isl_ast_expr_op_or_else:
      return boolean(arg(0) != 0 || arg(1) != 0);
    case isl_ast_expr_op_max:
    case isl_ast_expr_op_min: {
      DynamicAPInt result = arg(0);
      for (unsigned i = 1; i < expr.args.size(); ++i)
        result = expr.op == isl_ast_expr_op_max ? std::max(result, arg(i))
                                                : std::min(result, arg(i));
      return result;
    }
    case isl_ast_expr_op_minus:
      return -arg(0);
    case isl_ast_expr_op_add:
      return arg(0) + arg(1);
    case isl_ast_expr_op_sub:
      return arg(0) - arg(1);
    case isl_ast_expr_op_mul:
      return arg(0) * arg(1);
    // div is exact, and pdiv_q/pdiv_r have a non-negative dividend, so floor
    // division is correct for all three.
    case isl_ast_expr_op_div:
    case isl_ast_expr_op_fdiv_q:
    case isl_ast_expr_op_pdiv_q:
      return floorDiv(arg(0), arg(1));
    case isl_ast_expr_op_pdiv_r:
      return mod(arg(0), arg(1));
    case isl_ast_expr_op_zdiv_r:
      return arg(0) % arg(1);
    case isl_ast_expr_op_cond:
    case isl_ast_expr_op_select:
      return arg(0) != 0 ? arg(1) : arg(2);
    case isl_ast_expr_op_eq:
      return boolean(arg(0) == arg(1));
    case isl_ast_expr_op_le:
      return boolean(arg(0) <= arg(1));
    case isl_ast_expr_op_lt:
      return boolean(arg(0) < arg(1));
    case isl_ast_expr_op_ge:
      return boolean(arg(0) >= arg(1));
    case isl_ast_expr_op_gt:
      return boolean(arg(0) > arg(1));
    default:
      llvm::report_fatal_error("forEachPoint: unexpected isl AST operation");
  }
}

void run(const ScanNode& node, SmallVectorImpl<DynamicAPInt>& iterators,
         function_ref<void(ArrayRef<DynamicAPInt>)> visit,
         function_ref<bool()> shouldStop) {
  if (shouldStop && shouldStop()) return;
  switch (node.kind) {
    case ScanNode::kFor:
      for (iterators[node.iterator] = evaluate(node.init, iterators);
           evaluate(node.cond, iterators) != 0;
           iterators[node.iterator] += evaluate(node.inc, iterators)) {
        run(node.children[0], iterators, visit, shouldStop);
        if (shouldStop && shouldStop()) return;
      }
      return;
    case ScanNode::kIf:
      if (evaluate(node.cond, iterators) != 0)
        run(node.children[0], iterators, visit, shouldStop);
      else if (node.children.size() > 1)
        run(node.children[1], iterators, visit, shouldStop);
      return;
    case ScanNode::kBlock:
      for (const ScanNode& child : node.children) {
        run(child, iterators, visit, shouldStop);
        if (shouldStop && shouldStop()) return;
      }
      return;
    case ScanNode::kUser: {
      SmallVector<DynamicAPInt, 16> args;
      for (const ScanExpr& expr : node.args)
        args.push_back(evaluate(expr, iterators));
      visit(args);
      return;
    }
  }
}

}  // namespace

void forEachPoint(const presburger::IntegerRelation& rel,
                  function_ref<void(ArrayRef<int64_t>, ArrayRef<int64_t>)> fn,
                  function_ref<bool()> shouldStop) {
  unsigned numDomain = rel.getNumDomainVars();
  unsigned numVars = numDomain + rel.getNumRangeVars();

  // Lifting turns every local of the relation into an ordinary set coordinate,
  // placed after the domain and range ones, so isl builds the scan without
  // first eliminating the locals (isl_map_compute_divs). The schedule is the
  // identity, so the AST visits the lifted points in lexicographic order and
  // the witnesses of one point of `rel` arrive consecutively.
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_set* lifted = isl_basic_set_flatten(isl_basic_set_lift(
      isl_basic_map_wrap(convertRelationToBasicMap(rel, ctx))));
  lifted = isl_basic_set_set_tuple_name(lifted, "S");
  isl_map* identity = isl_set_identity(isl_set_from_basic_set(lifted));
  identity = isl_map_reset_tuple_id(identity, isl_dim_out);
  isl_ast_node* tree = buildAst(isl_union_map_from_map(identity), ctx);
  if (!tree) llvm::report_fatal_error("forEachPoint: isl codegen failed");
  AstScanner scanner;
  ScanNode root = scanner.buildNode(tree);
  isl_ast_node_free(tree);
  isl_ctx_free(ctx);

  SmallVector<DynamicAPInt> iterators(scanner.numIterators());
  SmallVector<int64_t> point(numVars), previous;
  run(
      root, iterators,
      [&](ArrayRef<DynamicAPInt> witness) {
        // The relation's constraints have int64_t coefficients and bound every
        // variable, so its points fit in int64_t.
        for (unsigned i = 0; i < numVars; ++i) point[i] = int64_t(witness[i]);
        if (!previous.empty() && previous == point) return;
        previous = point;
        fn(ArrayRef<int64_t>(point).take_front(numDomain),
           ArrayRef<int64_t>(point).drop_front(numDomain));
      },
      shouldStop);
}

FailureOr<isl_ast_node*> generateLoopNest(
    const presburger::IntegerRelation& rel, __isl_keep isl_ctx* ctx,
    ArrayRef<int> domainIndicesToSchedule) {
  isl_ast_node* tree = constructAst(rel, ctx, domainIndicesToSchedule);
  if (!tree) {
    return failure();
  }
  return tree;
}

FailureOr<std::string> generateLoopNestAsCStr(
    const presburger::IntegerRelation& rel,
    ArrayRef<int> domainIndicesToSchedule) {
  isl_ctx* ctx = isl_ctx_alloc();
  auto result = generateLoopNest(rel, ctx, domainIndicesToSchedule);
  if (failed(result)) {
    isl_ctx_free(ctx);
    return failure();
  }
  isl_ast_node* tree = result.value();
  char* cStr = isl_ast_node_to_C_str(tree);
  std::string actual = std::string(cStr);
  free(cStr);
  isl_ast_node_free(tree);
  isl_ctx_free(ctx);
  // Add a leading newline for ease of comparison with multiline strings.
  return actual.insert(0, "\n");
}

Value buildIslExpr(isl_ast_expr* expr, std::map<std::string, Value> ivToValue,
                   ImplicitLocOpBuilder& b,
                   const std::function<void(Operation*)>& createdOpCallback) {
  // The four types of isl expressions are:
  // - id: an identifier, e.g., "c0" (i.e. a loop index variable)
  // - int: a constant value, e.g., 42
  // - op: an arithmetic operation, e.g., x + y
  // - error: an invalid expression
  switch (isl_ast_expr_get_type(expr)) {
    case isl_ast_expr_id: {
      isl_id* id = isl_ast_expr_get_id(expr);
      std::string computationName(isl_id_get_name(id));
      isl_id_free(id);
      return ivToValue[computationName];
    }
    case isl_ast_expr_int: {
      isl_val* val = isl_ast_expr_get_val(expr);
      auto intValue = isl_val_get_num_si(val);
      isl_val_free(val);
      auto op = arith::ConstantIntOp::create(b, intValue, 32);
      createdOpCallback(op);
      return op->getResult(0);
    }
    case isl_ast_expr_op: {
      // ISL operations types are defined in
      // https://github.com/j2kun/isl/blob/58eaa613ffcd2b11c36a773c19531f75fbcfc66c/include/isl/ast_type.h#L16
      enum isl_ast_expr_op_type type = isl_ast_expr_get_op_type(expr);
      auto getArgs = [&](isl_ast_expr* expr) {
        SmallVector<Value> args;
        for (int i = 0; i < isl_ast_expr_get_op_n_arg(expr); ++i) {
          auto* arg = isl_ast_expr_get_op_arg(expr, i);
          auto argValue = buildIslExpr(arg, ivToValue, b, createdOpCallback);
          args.push_back(argValue);
          isl_ast_expr_free(arg);
        }
        return args;
      };
      if (islBinaryOpToMlir.contains(type)) {
        // Binary ops with 1-1 correspondence
        auto mlirOp = islBinaryOpToMlir[isl_ast_expr_get_op_type(expr)];
        SmallVector<Value> args = getArgs(expr);
        if (args.size() > 2) {
          // Reduce variadic ops to binary ops
          Value accum = args[0];
          for (size_t i = 1; i < args.size(); ++i) {
            auto* op = b.create(OperationState(
                b.getLoc(), mlirOp, {accum, args[i]}, {accum.getType()}));
            createdOpCallback(op);
            accum = op->getResult(0);
          }
          return accum;
        }

        auto* op = b.create(
            OperationState(b.getLoc(), mlirOp, args, {args[0].getType()}));
        createdOpCallback(op);
        return op->getResult(0);
      }

      if (type == isl_ast_op_minus) {
        // Unary op
        SmallVector<Value> args = getArgs(expr);
        auto op = arith::SubIOp::create(
            b, b.getLoc(), arith::ConstantIntOp::create(b, 0, 32), args[0]);
        createdOpCallback(op);
        return op->getResult(0);
      }

      if (islCmpToMlirAttr.contains(type)) {
        // Comparison ops
        SmallVector<Value> args = getArgs(expr);

        // Sometimes the result of a cmpi op is subsequently checked to be 0 or
        // 1, so we have to be careful to ensure the argument types are the
        // correct widths and extend if not.
        if (!allTypesMatch(args)) {
          args = extendToCommonWidth(b, args, createdOpCallback);
        }

        // Keep the comparison result as an i1. ISL keeps boolean and integer
        // AST expressions distinct -- a comparison only ever feeds and/or
        // (logical), select, or an `if` guard, never integer arithmetic -- so
        // there is no need to widen it to an index 0/1.
        auto op =
            arith::CmpIOp::create(b, islCmpToMlirAttr[type], args[0], args[1]);
        createdOpCallback(op);
        return op->getResult(0);
      } else if (type == isl_ast_op_select) {
        // Select op. The condition is a boolean expression, so it is already an
        // i1; an index-typed condition from another path is still cast.
        SmallVector<Value> args = getArgs(expr);
        Value cond = args[0];
        if (!cond.getType().isInteger(1)) {
          auto condI1 = arith::IndexCastOp::create(b, b.getI1Type(), cond);
          createdOpCallback(condI1);
          cond = condI1->getResult(0);
        }
        auto op = arith::SelectOp::create(b, cond, args[1], args[2]);
        createdOpCallback(op);
        return op->getResult(0);
      }

      if (type == isl_ast_expr_op_zdiv_r) {
        // Remainder op
        SmallVector<Value> args = getArgs(expr);
        auto op = arith::RemSIOp::create(b, args[0], args[1]);
        createdOpCallback(op);
        return op->getResult(0);
      }
      if (type == isl_ast_op_pdiv_r) {
        // pdiv_r(a, b) = a - b * floordiv(a, b)
        SmallVector<Value> args = getArgs(expr);
        Value a = args[0];
        Value bVal = args[1];

        auto div = arith::FloorDivSIOp::create(b, a, bVal);
        auto mul = arith::MulIOp::create(b, bVal, div);
        auto sub = arith::SubIOp::create(b, a, mul);

        createdOpCallback(div);
        createdOpCallback(mul);
        createdOpCallback(sub);
        return sub->getResult(0);
      }
      isl_ast_expr_op_type opType = isl_ast_expr_get_op_type(expr);
      char* cStr = isl_ast_expr_to_C_str(expr);
      emitError(b.getLoc())
          << "Unsupported ISL operation type " << int(opType) << ": " << cStr;
      free(cStr);
      return Value();
    }
    case isl_ast_expr_error: {
      emitError(b.getLoc())
          << "ISL ast expr error: " << isl_ast_expr_to_str(expr);
      return Value();
    }
  }
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNodeFor(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder_);
  // Collect the string names of the iterators
  isl_ast_expr* iter = isl_ast_node_for_get_iterator(node);
  isl_id* identifier = isl_ast_expr_get_id(iter);
  std::string nameStr(isl_id_get_name(identifier));
  isl_id_free(identifier);
  isl_ast_expr_free(iter);

  Value oldIv = nullptr;
  auto it = ivToValue_.find(nameStr);
  if (it != ivToValue_.end()) {
    oldIv = it->second;
  }

  isl_ast_expr* init = isl_ast_node_for_get_init(node);
  Value lbInt = buildIslExpr(init, ivToValue_, builder_, createdOpCallback_);
  isl_ast_expr_free(init);

  // The upper bound is a less than equal or less than operation in the AST
  // codegen.
  isl_ast_expr* cond = isl_ast_node_for_get_cond(node);
  assert(isl_ast_expr_get_type(cond) == isl_ast_expr_op &&
         "expected condition to be an operation");
  isl_ast_expr_op_type condOpType = isl_ast_expr_get_op_type(cond);
  assert((condOpType == isl_ast_op_le || condOpType == isl_ast_op_lt) &&
         "expected upper bound to be a less than or less than equal operation");
  isl_ast_expr* condUpper = isl_ast_expr_get_op_arg(cond, 1);
  Value ubVal =
      buildIslExpr(condUpper, ivToValue_, builder_, createdOpCallback_);
  if (condOpType == isl_ast_op_le) {
    ubVal = arith::AddIOp::create(
        builder_, currentLoc_, ubVal,
        arith::ConstantIntOp::create(builder_, currentLoc_, 1, 32));
    createdOpCallback_(ubVal.getDefiningOp());
  }
  isl_ast_expr_free(condUpper);
  isl_ast_expr_free(cond);

  isl_ast_expr* step = isl_ast_node_for_get_inc(node);
  Value stepInt = buildIslExpr(step, ivToValue_, builder_, createdOpCallback_);
  isl_ast_expr_free(step);

  // Create the scf for loop
  auto loop = scf::ForOp::create(
      builder_, currentLoc_, lbInt, ubVal, stepInt, currentIterArgs_,
      [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iv,
          ValueRange args) {
        Value ivCast = iv;
        if (llvm::isa<mlir::IndexType>(iv.getType())) {
          ivCast = arith::IndexCastOp::create(nestedBuilder, nestedLoc,
                                              nestedBuilder.getI32Type(), iv);
        }
        if (reverse_) {
          // To reverse a loop running `iv` from `lbInt` to `ubVal` with
          // `stepInt`: The forward loop runs with iv = lbInt + k * stepInt. The
          // backwards loop runs with
          //   actualIv = lbInt + (max_k - k) * stepInt
          // where max_k = (ubVal - 1 - lbInt) // stepInt
          //
          // Solving for actualIv:
          //   actualIv = (lbInt + max_k * stepInt) - k * stepInt
          //            = actualUb - (iv - lbInt)
          // where actualUb = lbInt + max_k * stepInt.
          Value constOne =
              arith::ConstantIntOp::create(nestedBuilder, nestedLoc, 1, 32);
          Value ubMinusOne =
              arith::SubIOp::create(nestedBuilder, nestedLoc, ubVal, constOne);
          Value tmp = arith::SubIOp::create(nestedBuilder, nestedLoc,
                                            ubMinusOne, lbInt);
          Value maxK = arith::FloorDivSIOp::create(nestedBuilder, nestedLoc,
                                                   tmp, stepInt);
          Value maxKTimesStep =
              arith::MulIOp::create(nestedBuilder, nestedLoc, maxK, stepInt);
          Value actualUb = arith::AddIOp::create(nestedBuilder, nestedLoc,
                                                 lbInt, maxKTimesStep);
          Value ivMinusLb =
              arith::SubIOp::create(nestedBuilder, nestedLoc, ivCast, lbInt);
          Value actualIv = arith::SubIOp::create(nestedBuilder, nestedLoc,
                                                 actualUb, ivMinusLb);

          createdOpCallback_(ubMinusOne.getDefiningOp());
          createdOpCallback_(tmp.getDefiningOp());
          createdOpCallback_(maxK.getDefiningOp());
          createdOpCallback_(maxKTimesStep.getDefiningOp());
          createdOpCallback_(actualUb.getDefiningOp());
          createdOpCallback_(ivMinusLb.getDefiningOp());
          createdOpCallback_(actualIv.getDefiningOp());

          ivToValue_.insert(std::make_pair(nameStr, actualIv));
        } else {
          ivToValue_.insert(std::make_pair(nameStr, ivCast));
        }
        // It is safe to store ValueRange args because it points to block
        // arguments of a loop operation that we also own.
        currentIterArgs_ = args;
        currentLoc_ = nestedLoc;
      });
  createdOpCallback_(loop);

  builder_.setInsertionPointToEnd(loop.getBody());
  loops_.push_back(loop);

  isl_ast_node* tmp = isl_ast_node_for_get_body(node);
  auto visitBody = visitAstNode(tmp, bodyBuilder);
  if (failed(visitBody)) {
    if (oldIv) {
      ivToValue_[nameStr] = oldIv;
    } else {
      ivToValue_.erase(nameStr);
    }
    return failure();
  }
  isl_ast_node_free(tmp);

  // Yield the results of the body.
  if (!visitBody.value().empty()) {
    builder_.setInsertionPointToEnd(loop.getBody());
    auto yieldOp =
        scf::YieldOp::create(builder_, currentLoc_, visitBody.value());
    createdOpCallback_(yieldOp);
  }

  if (oldIv) {
    ivToValue_[nameStr] = oldIv;
  } else {
    ivToValue_.erase(nameStr);
  }

  // Return the results of the body.
  return scf::ValueVector(loop.getResults());
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNodeIf(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder_);
  isl_ast_expr* cond = isl_ast_node_if_get_cond(node);
  Value condVal = buildIslExpr(cond, ivToValue_, builder_, createdOpCallback_);
  isl_ast_expr_free(cond);

  // The else branch is a no-op that passes the incoming iter args through
  // unchanged. We must snapshot them now, before visiting the then branch:
  // visiting a nested for loop overwrites currentIterArgs_ to point at that
  // inner loop's block arguments (see visitAstNodeFor), which live in the then
  // region and would not dominate a yield in the else region.
  SmallVector<Value> incomingIterArgs(currentIterArgs_.begin(),
                                      currentIterArgs_.end());

  Value condValI1 = condVal;
  if (!condValI1.getType().isInteger(1)) {
    auto cast =
        arith::IndexCastOp::create(builder_, builder_.getI1Type(), condVal);
    createdOpCallback_(cast);
    condValI1 = cast->getResult(0);
  }
  auto ifOp = scf::IfOp::create(builder_, currentLoc_,
                                TypeRange(incomingIterArgs), condValI1,
                                /*addThenBlock=*/true, /*addElseBlock=*/true);
  createdOpCallback_(ifOp);

  isl_ast_node* thenNode = isl_ast_node_if_get_then_node(node);
  builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto visitThenBody = visitAstNode(thenNode, bodyBuilder);
  isl_ast_node_free(thenNode);
  if (failed(visitThenBody)) {
    return failure();
  }

  isl_ast_node* elseNode = isl_ast_node_if_get_else_node(node);
  bool hasElse = (elseNode != nullptr);
  FailureOr<scf::ValueVector> visitElseBody;
  if (hasElse) {
    currentIterArgs_ = incomingIterArgs;
    builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
    visitElseBody = visitAstNode(elseNode, bodyBuilder);
    isl_ast_node_free(elseNode);
    if (failed(visitElseBody)) {
      return failure();
    }
  } else {
    isl_ast_node_free(elseNode);
  }

  // For all if statements but the innermost, yield the results of the nested
  // ifs
  if (!visitThenBody.value().empty()) {
    builder_.setInsertionPointToEnd(&ifOp.getThenRegion().front());
    auto yieldOp =
        scf::YieldOp::create(builder_, currentLoc_, visitThenBody.value());
    createdOpCallback_(yieldOp);
    builder_.setInsertionPointToEnd(&ifOp.getElseRegion().front());
    if (hasElse) {
      assert(visitElseBody.value().size() == visitThenBody.value().size() &&
             "then and else branches must yield the same number of values");
      yieldOp =
          scf::YieldOp::create(builder_, currentLoc_, visitElseBody.value());
    } else {
      yieldOp = scf::YieldOp::create(builder_, currentLoc_, incomingIterArgs);
    }
    createdOpCallback_(yieldOp);
  }

  // Return the results of the body.
  return scf::ValueVector(ifOp.getResults());
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNodeUser(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  isl_ast_expr* expr = isl_ast_node_user_get_expr(node);

  SmallVector<Value> exprs;
  // expr 0 is the statement S
  for (int i = 1; i < isl_ast_expr_get_op_n_arg(expr); i++) {
    isl_ast_expr* arg = isl_ast_expr_get_op_arg(expr, i);
    Value value = buildIslExpr(arg, ivToValue_, builder_, createdOpCallback_);
    exprs.push_back(value);
    isl_ast_expr_free(arg);
  }

  // In the body of the innermost loop, call the body building function using
  // the generated indexing expressions and yield its results.
  scf::ValueVector results =
      bodyBuilder(builder_, currentLoc_, exprs, currentIterArgs_);
  assert(results.size() == currentIterArgs_.size() &&
         "loop nest body must return as many values as loop has iteration "
         "arguments");
  isl_ast_expr_free(expr);

  return results;
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNodeBlock(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  isl_ast_node_list* list = isl_ast_node_block_get_children(node);
  int n = isl_ast_node_list_n_ast_node(list);
  scf::ValueVector results = currentIterArgs_;
  for (int i = 0; i < n; ++i) {
    isl_ast_node* child = isl_ast_node_list_get_ast_node(list, i);
    currentIterArgs_ = results;
    auto childResults = visitAstNode(child, bodyBuilder);
    isl_ast_node_free(child);
    if (failed(childResults)) {
      isl_ast_node_list_free(list);
      return failure();
    }
    results = childResults.value();
  }
  isl_ast_node_list_free(list);
  return results;
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNode(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  switch (isl_ast_node_get_type(node)) {
    case isl_ast_node_for:
      return visitAstNodeFor(node, bodyBuilder);
    case isl_ast_node_if:
      return visitAstNodeIf(node, bodyBuilder);
    case isl_ast_node_block:
      return visitAstNodeBlock(node, bodyBuilder);
    case isl_ast_node_user:
      return visitAstNodeUser(node, bodyBuilder);
    default:
      char* cStr = isl_ast_node_to_C_str(node);
      emitError(builder_.getLoc()) << "unhandled ISL node type: " << cStr;
      free(cStr);
      return failure();
  }
}

FailureOr<scf::ForOp> MLIRLoopNestGenerator::generateForLoop(
    const presburger::IntegerRelation& rel, ValueRange initArgs,
    BodyBuilderFn bodyBuilder, ArrayRef<int> domainIndicesToSchedule,
    bool reverse) {
  OpBuilder::InsertionGuard guard(builder_);
  reverse_ = reverse;
  isl_ast_node* tree = constructAst(rel, ctx_, domainIndicesToSchedule);
  if (!tree) {
    return failure();
  }

  isl_ast_node* node = tree;
  currentIterArgs_ = initArgs;

  auto result = visitAstNode(node, bodyBuilder);
  if (failed(result)) {
    return failure();
  }
  isl_ast_node_free(node);

  return loops_.front();
}

}  // namespace heir
}  // namespace mlir
