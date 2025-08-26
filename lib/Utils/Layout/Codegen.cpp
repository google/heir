#include "lib/Utils/Layout/Codegen.h"

#include <cassert>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

#include "lib/Utils/Layout/IslConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"             // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"     // from @llvm-project
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
    {isl_ast_op_add, "arith.addi"},     {isl_ast_op_sub, "arith.subi"},
    {isl_ast_op_mul, "arith.muli"},     {isl_ast_op_div, "arith.divsi"},
    {isl_ast_op_pdiv_r, "arith.remsi"}, {isl_ast_op_pdiv_q, "arith.divsi"},
    {isl_ast_op_max, "arith.maxsi"},    {isl_ast_op_min, "arith.minsi"},
    {isl_ast_op_and, "arith.andi"},     {isl_ast_op_or, "arith.ori"},
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
static __isl_give isl_union_map* createSchedule(__isl_keep isl_basic_map* bmap,
                                                isl_ctx* ctx) {
  isl_basic_map* schedule_bmap = isl_basic_map_copy(bmap);
  unsigned numIn = isl_basic_map_dim(schedule_bmap, isl_dim_in);
  unsigned numOut = isl_basic_map_dim(schedule_bmap, isl_dim_out);

  // Insert two new dimensions for the original range variables into the domain.
  schedule_bmap =
      isl_basic_map_insert_dims(schedule_bmap, isl_dim_in, numIn, numOut);

  // Add constraints to equate the new domain dimensions with the original range
  // dimensions.
  for (unsigned i = 0; i < numOut; ++i) {
    isl_constraint* c = isl_constraint_alloc_equality(
        isl_basic_map_get_local_space(schedule_bmap));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, numIn + i, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
    schedule_bmap = isl_basic_map_add_constraint(schedule_bmap, c);
  }

  // Set the domain tuple name to "S". This will be used in codegen for the
  // statement to be executed, e.g., S(a, b, c, d)
  schedule_bmap = isl_basic_map_set_tuple_name(schedule_bmap, isl_dim_in, "S");
  return isl_union_map_from_basic_map(schedule_bmap);
}

__isl_give isl_ast_node* constructAst(const presburger::IntegerRelation& rel,
                                      isl_ctx* ctx) {
  isl_basic_map* bmap = convertRelationToBasicMap(rel, ctx);
  isl_union_map* schedule = createSchedule(bmap, ctx);
  isl_basic_map_free(bmap);

  // Context and options are intentionally empty. We don't need any of these
  // features, though I admit I have not looked into what they can provide.
  isl_set* context = isl_set_universe(isl_space_params_alloc(ctx, 0));
  isl_union_map* options = isl_union_map_empty(isl_space_params_alloc(ctx, 0));

  // Build the AST
  isl_ast_build* build = isl_ast_build_from_context(context);
  build = isl_ast_build_set_options(build, options);
  isl_ast_node* tree = isl_ast_build_node_from_schedule_map(build, schedule);
  isl_ast_build_free(build);

  return tree;
}

FailureOr<isl_ast_node*> generateLoopNest(
    const presburger::IntegerRelation& rel, __isl_keep isl_ctx* ctx) {
  isl_ast_node* tree = constructAst(rel, ctx);
  if (!tree) {
    return failure();
  }
  return tree;
}

FailureOr<std::string> generateLoopNestAsCStr(
    const presburger::IntegerRelation& rel) {
  isl_ctx* ctx = isl_ctx_alloc();
  auto result = generateLoopNest(rel, ctx);
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
                   ImplicitLocOpBuilder& b) {
  // The four types of isl expressions are:
  // - id: an identifier, e.g., "c0" (i.e. a loop index variable)
  // - int: a constant value, e.g., 42
  // - op: an arithmetic operation, e.g., x + y
  // - error: an invalid expression
  switch (isl_ast_expr_get_type(expr)) {
    case isl_ast_expr_id: {
      isl_id* id = isl_ast_expr_get_id(expr);
      std::string computation_name(isl_id_get_name(id));
      isl_id_free(id);
      return ivToValue[computation_name];
    }
    case isl_ast_expr_int: {
      isl_val* val = isl_ast_expr_get_val(expr);
      auto int_value = isl_val_get_num_si(val);
      isl_val_free(val);
      return b.create<arith::ConstantIndexOp>(int_value);
    }
    case isl_ast_expr_op: {
      // ISL operations types are defined in
      // https://github.com/j2kun/isl/blob/58eaa613ffcd2b11c36a773c19531f75fbcfc66c/include/isl/ast_type.h#L16
      enum isl_ast_expr_op_type type = isl_ast_expr_get_op_type(expr);
      auto getArgs = [&](isl_ast_expr* expr) {
        SmallVector<Value> args;
        for (int i = 0; i < isl_ast_expr_get_op_n_arg(expr); ++i) {
          auto arg = isl_ast_expr_get_op_arg(expr, i);
          auto argValue = buildIslExpr(arg, ivToValue, b);
          args.push_back(argValue);
          isl_ast_expr_free(arg);
        }
        return args;
      };
      if (islBinaryOpToMlir.contains(type)) {
        // Binary ops
        auto mlirOp = islBinaryOpToMlir[isl_ast_expr_get_op_type(expr)];
        SmallVector<Value> args = getArgs(expr);
        auto op = b.create(
            OperationState(b.getLoc(), mlirOp, args, {args[0].getType()}));
        return op->getResult(0);
      } else if (type == isl_ast_op_minus) {
        // Unary op
        SmallVector<Value> args = getArgs(expr);
        auto op = b.create<arith::SubIOp>(
            b.getLoc(), b.create<arith::ConstantIndexOp>(0), args[0]);
        return op->getResult(0);
      } else if (islCmpToMlirAttr.contains(type)) {
        // Comparison ops
        SmallVector<Value> args = getArgs(expr);
        auto op =
            arith::CmpIOp::create(b, islCmpToMlirAttr[type], args[0], args[1]);
        return op->getResult(0);
      }
      char* cStr = isl_ast_expr_to_C_str(expr);
      emitError(b.getLoc()) << "Unsupported ISL operation: " << cStr;
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

FailureOr<scf::ForOp> MLIRLoopNestGenerator::generateForLoop(
    const presburger::IntegerRelation& rel, ValueRange initArgs,
    function_ref<scf::ValueVector(OpBuilder&, Location, ValueRange, ValueRange)>
        bodyBuilder) {
  isl_ast_node* tree = constructAst(rel, ctx_);
  if (!tree) {
    return failure();
  }

  isl_ast_node* node = tree;
  ValueRange currentIterArgs = initArgs;
  Location currentLoc = builder_.getUnknownLoc();
  std::map<std::string, Value> ivToValue;
  SmallVector<scf::ForOp> loops;
  // Create the loop nest structure
  while (isl_ast_node_get_type(node) == isl_ast_node_for) {
    // Collect the string names of the iterators
    isl_ast_expr* iter = isl_ast_node_for_get_iterator(node);
    isl_id* identifier = isl_ast_expr_get_id(iter);
    std::string name_str(isl_id_get_name(identifier));
    isl_id_free(identifier);
    isl_ast_expr_free(iter);

    isl_ast_expr* init = isl_ast_node_for_get_init(node);
    Value lbInt = buildIslExpr(init, ivToValue, builder_);
    isl_ast_expr_free(init);

    // The upper bound is always a less than equal operation in the AST codegen.
    isl_ast_expr* cond = isl_ast_node_for_get_cond(node);
    assert((isl_ast_expr_get_type(cond) == isl_ast_expr_op &&
            isl_ast_expr_get_op_type(cond) == isl_ast_op_le) &&
           "expected upper bound to be a less than equal operation");
    isl_ast_expr* cond_upper = isl_ast_expr_get_op_arg(cond, 1);
    Value ubVal = builder_.create<arith::AddIOp>(
        currentLoc, buildIslExpr(cond_upper, ivToValue, builder_),
        builder_.create<arith::ConstantIndexOp>(currentLoc, 1));
    isl_ast_expr_free(cond_upper);
    isl_ast_expr_free(cond);

    isl_ast_expr* step = isl_ast_node_for_get_inc(node);
    Value stepInt = buildIslExpr(step, ivToValue, builder_);
    isl_ast_expr_free(step);

    // Create the scf for loop
    auto loop = scf::ForOp::create(
        builder_, currentLoc, lbInt, ubVal, stepInt, currentIterArgs,
        [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iv,
            ValueRange args) {
          ivToValue.insert(std::make_pair(name_str, iv));
          // It is safe to store ValueRange args because it points to block
          // arguments of a loop operation that we also own.
          currentIterArgs = args;
          currentLoc = nestedLoc;
        });

    // Set the builder to point to the body of the newly created loop. We don't
    // do this in the callback because the builder is reset when the callback
    // returns.
    builder_.setInsertionPointToStart(loop.getBody());
    loops.push_back(loop);

    node = isl_ast_node_for_get_body(node);
  }

  // For all loops but the innermost, yield the results of the nested loop.
  for (unsigned i = 0, e = loops.size() - 1; i < e; ++i) {
    builder_.setInsertionPointToEnd(loops[i].getBody());
    scf::YieldOp::create(builder_, currentLoc, loops[i + 1].getResults());
  }

  // Handle any conditionals on the qualifiers.
  Block* innerBlock = loops.back().getBody();
  SmallVector<scf::IfOp> ifOps;
  while (isl_ast_node_get_type(node) == isl_ast_node_if) {
    builder_.setInsertionPointToStart(innerBlock);

    isl_ast_expr* cond = isl_ast_node_if_get_cond(node);
    Value condVal = buildIslExpr(cond, ivToValue, builder_);
    isl_ast_expr_free(cond);

    // Build scf if operation with the result types of the iter args
    auto ifOp = scf::IfOp::create(
        builder_, currentLoc, TypeRange(loops.back().getRegionIterArgs()),
        condVal, /*addThenBlock=*/true, /*addElseBlock=*/true);

    // TODO:(#2120): Handle ISL else conditions.
    isl_ast_node* elseNode = isl_ast_node_if_get_else_node(node);
    assert(elseNode == nullptr && "expected no else conditions");
    isl_ast_node_free(elseNode);

    node = isl_ast_node_if_get_then_node(node);
    innerBlock = &ifOp.getThenRegion().front();
    ifOps.push_back(ifOp);
  }

  // For all if statements but the innermost, yield the results of the nested
  // ifs
  if (!ifOps.empty()) {
    for (unsigned i = 0, e = ifOps.size() - 1; i < e; ++i) {
      builder_.setInsertionPointToEnd(&ifOps[i].getThenRegion().front());
      scf::YieldOp::create(builder_, currentLoc, ifOps[i + 1].getResults());
      builder_.setInsertionPointToEnd(&ifOps[i].getElseRegion().front());
      scf::YieldOp::create(builder_, currentLoc, ifOps[i + 1].getResults());
    }
  }

  // The body of the innermost loop should contain the user statement S.
  if (isl_ast_node_get_type(node) != isl_ast_node_user) {
    char* cStr = isl_ast_node_to_C_str(node);
    emitError(builder_.getLoc()) << "unhandled ISL node type: " << cStr;
    free(cStr);
    return failure();
  }

  // Generate the indexing expressions inside the inner block.
  builder_.setInsertionPointToStart(innerBlock);
  isl_ast_expr* expr = isl_ast_node_user_get_expr(node);
  assert(expr != nullptr && "expected pure loop nest with only user statement");

  SmallVector<Value> exprs;
  // expr 0 is the statement S
  for (int i = 1; i < isl_ast_expr_get_op_n_arg(expr); i++) {
    isl_ast_expr* arg = isl_ast_expr_get_op_arg(expr, i);
    Value value = buildIslExpr(arg, ivToValue, builder_);
    exprs.push_back(value);
    isl_ast_expr_free(arg);
  }

  // In the body of the innermost loop, call the body building function using
  // the generated indexing expressions and yield its results.
  builder_.setInsertionPointToEnd(innerBlock);
  scf::ValueVector results = bodyBuilder(builder_, currentLoc, exprs,
                                         loops.back().getRegionIterArgs());
  assert(results.size() == initArgs.size() &&
         "loop nest body must return as many values as loop has iteration "
         "arguments");

  if (!ifOps.empty()) {
    // Yield results of the body builder from the inner most if statement.
    builder_.setInsertionPointToEnd(&ifOps.back().getThenRegion().front());
    scf::YieldOp::create(builder_, currentLoc, results);
    builder_.setInsertionPointToEnd(&ifOps.back().getElseRegion().front());
    scf::YieldOp::create(builder_, currentLoc,
                         loops.back().getRegionIterArgs());
    results = ifOps.front().getResults();
  }

  // Yield from the innermost loop. These will either be the results from the
  // outermost if statements, or the results of the body builder.
  builder_.setInsertionPointToEnd(loops.back().getBody());
  scf::YieldOp::create(builder_, currentLoc, results);

  isl_ast_expr_free(expr);
  isl_ast_node_free(node);
  isl_ast_node_free(tree);

  return loops[0];
}

}  // namespace heir
}  // namespace mlir
