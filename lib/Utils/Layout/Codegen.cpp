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
    {isl_ast_op_add, "arith.addi"},          {isl_ast_op_sub, "arith.subi"},
    {isl_ast_op_mul, "arith.muli"},          {isl_ast_op_div, "arith.divsi"},
    {isl_ast_op_pdiv_r, "arith.remsi"},      {isl_ast_op_pdiv_q, "arith.divsi"},
    {isl_ast_op_max, "arith.maxsi"},         {isl_ast_op_min, "arith.minsi"},
    {isl_ast_op_and, "arith.andi"},          {isl_ast_op_or, "arith.ori"},
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
static __isl_give isl_union_map* createSchedule(__isl_keep isl_basic_map* bmap,
                                                isl_ctx* ctx) {
  isl_basic_map* scheduleBmap = isl_basic_map_copy(bmap);
  unsigned numIn = isl_basic_map_dim(scheduleBmap, isl_dim_in);
  unsigned numOut = isl_basic_map_dim(scheduleBmap, isl_dim_out);

  // Insert two new dimensions for the original range variables into the domain.
  scheduleBmap =
      isl_basic_map_insert_dims(scheduleBmap, isl_dim_in, numIn, numOut);

  // Add constraints to equate the new domain dimensions with the original range
  // dimensions.
  for (unsigned i = 0; i < numOut; ++i) {
    isl_constraint* c = isl_constraint_alloc_equality(
        isl_basic_map_get_local_space(scheduleBmap));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, numIn + i, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
    scheduleBmap = isl_basic_map_add_constraint(scheduleBmap, c);
  }

  // Set the domain tuple name to "S". This will be used in codegen for the
  // statement to be executed, e.g., S(a, b, c, d)
  scheduleBmap = isl_basic_map_set_tuple_name(scheduleBmap, isl_dim_in, "S");
  return isl_union_map_from_basic_map(scheduleBmap);
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
      std::string computationName(isl_id_get_name(id));
      isl_id_free(id);
      return ivToValue[computationName];
    }
    case isl_ast_expr_int: {
      isl_val* val = isl_ast_expr_get_val(expr);
      auto intValue = isl_val_get_num_si(val);
      isl_val_free(val);
      return arith::ConstantIntOp::create(b, intValue, 32);
    }
    case isl_ast_expr_op: {
      // ISL operations types are defined in
      // https://github.com/j2kun/isl/blob/58eaa613ffcd2b11c36a773c19531f75fbcfc66c/include/isl/ast_type.h#L16
      enum isl_ast_expr_op_type type = isl_ast_expr_get_op_type(expr);
      auto getArgs = [&](isl_ast_expr* expr) {
        SmallVector<Value> args;
        for (int i = 0; i < isl_ast_expr_get_op_n_arg(expr); ++i) {
          auto* arg = isl_ast_expr_get_op_arg(expr, i);
          auto argValue = buildIslExpr(arg, ivToValue, b);
          args.push_back(argValue);
          isl_ast_expr_free(arg);
        }
        return args;
      };
      if (islBinaryOpToMlir.contains(type)) {
        // Binary ops with 1-1 correspondence
        auto mlirOp = islBinaryOpToMlir[isl_ast_expr_get_op_type(expr)];
        SmallVector<Value> args = getArgs(expr);
        auto* op = b.create(
            OperationState(b.getLoc(), mlirOp, args, {args[0].getType()}));
        return op->getResult(0);
      }

      if (type == isl_ast_op_minus) {
        // Unary op
        SmallVector<Value> args = getArgs(expr);
        auto op = arith::SubIOp::create(
            b, b.getLoc(), arith::ConstantIntOp::create(b, 0, 32), args[0]);
        return op->getResult(0);
      }

      if (islCmpToMlirAttr.contains(type)) {
        // Comparison ops
        SmallVector<Value> args = getArgs(expr);
        auto op =
            arith::CmpIOp::create(b, islCmpToMlirAttr[type], args[0], args[1]);
        return op->getResult(0);
      }

      if (type == isl_ast_op_select) {
        // Select op
        SmallVector<Value> args = getArgs(expr);
        auto op = arith::SelectOp::create(b, args[0], args[1], args[2]);
        return op->getResult(0);
      }

      if (type == isl_ast_expr_op_zdiv_r) {
        // Remainder op with comparison to zero
        SmallVector<Value> args = getArgs(expr);
        auto op = arith::RemSIOp::create(b, args[0], args[1]);
        auto eqOp =
            arith::CmpIOp::create(b, arith::CmpIPredicate::eq, op,
                                  arith::ConstantIntOp::create(b, 0, 32));
        return eqOp->getResult(0);
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
  // Collect the string names of the iterators
  isl_ast_expr* iter = isl_ast_node_for_get_iterator(node);
  isl_id* identifier = isl_ast_expr_get_id(iter);
  std::string nameStr(isl_id_get_name(identifier));
  isl_id_free(identifier);
  isl_ast_expr_free(iter);

  isl_ast_expr* init = isl_ast_node_for_get_init(node);
  Value lbInt = buildIslExpr(init, ivToValue_, builder_);
  isl_ast_expr_free(init);

  // The upper bound is always a less than equal operation in the AST codegen.
  isl_ast_expr* cond = isl_ast_node_for_get_cond(node);
  assert((isl_ast_expr_get_type(cond) == isl_ast_expr_op &&
          isl_ast_expr_get_op_type(cond) == isl_ast_op_le) &&
         "expected upper bound to be a less than equal operation");
  isl_ast_expr* condUpper = isl_ast_expr_get_op_arg(cond, 1);
  Value ubVal = arith::AddIOp::create(
      builder_, currentLoc_, buildIslExpr(condUpper, ivToValue_, builder_),
      arith::ConstantIntOp::create(builder_, currentLoc_, 1, 32));
  isl_ast_expr_free(condUpper);
  isl_ast_expr_free(cond);

  isl_ast_expr* step = isl_ast_node_for_get_inc(node);
  Value stepInt = buildIslExpr(step, ivToValue_, builder_);
  isl_ast_expr_free(step);

  // Create the scf for loop
  auto loop = scf::ForOp::create(
      builder_, currentLoc_, lbInt, ubVal, stepInt, currentIterArgs_,
      [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iv,
          ValueRange args) {
        ivToValue_.insert(std::make_pair(nameStr, iv));
        // It is safe to store ValueRange args because it points to block
        // arguments of a loop operation that we also own.
        currentIterArgs_ = args;
        currentLoc_ = nestedLoc;
      });

  // Set the builder to point to the body of the newly created loop. We don't
  // do this in the callback because the builder is reset when the callback
  // returns.
  builder_.setInsertionPointToStart(loop.getBody());
  loops_.push_back(loop);

  isl_ast_node* tmp = isl_ast_node_for_get_body(node);
  builder_.setInsertionPointToStart(loop.getBody());
  auto visitBody = visitAstNode(tmp, bodyBuilder);
  if (failed(visitBody)) {
    return failure();
  }
  isl_ast_node_free(tmp);

  // Yield the results of the body.
  if (!visitBody.value().empty()) {
    builder_.setInsertionPointToEnd(loop.getBody());
    scf::YieldOp::create(builder_, currentLoc_, visitBody.value());
  }

  // Return the results of the body.
  return scf::ValueVector(loop.getResults());
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNodeIf(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  isl_ast_expr* cond = isl_ast_node_if_get_cond(node);
  Value condVal = buildIslExpr(cond, ivToValue_, builder_);
  isl_ast_expr_free(cond);

  // Build scf if operation with the result types of the iter args
  auto ifOp =
      scf::IfOp::create(builder_, currentLoc_, TypeRange(currentIterArgs_),
                        condVal, /*addThenBlock=*/true, /*addElseBlock=*/true);

  // TODO:(#2120): Handle ISL else conditions.
  isl_ast_node* elseNode = isl_ast_node_if_get_else_node(node);
  assert(elseNode == nullptr && "expected no else conditions");
  isl_ast_node_free(elseNode);

  isl_ast_node* tmp = isl_ast_node_if_get_then_node(node);
  builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto visitBody = visitAstNode(tmp, bodyBuilder);
  if (failed(visitBody)) {
    return failure();
  }
  isl_ast_node_free(tmp);

  // For all if statements but the innermost, yield the results of the nested
  // ifs
  if (!visitBody.value().empty()) {
    builder_.setInsertionPointToEnd(&ifOp.getThenRegion().front());
    scf::YieldOp::create(builder_, currentLoc_, visitBody.value());
    builder_.setInsertionPointToEnd(&ifOp.getElseRegion().front());
    scf::YieldOp::create(builder_, currentLoc_, currentIterArgs_);
  }

  // Return the results of the body.
  return scf::ValueVector(ifOp.getResults());
}

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNodeUser(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  // Generate the indexing expressions inside the inner block.
  isl_ast_expr* expr = isl_ast_node_user_get_expr(node);

  SmallVector<Value> exprs;
  // expr 0 is the statement S
  for (int i = 1; i < isl_ast_expr_get_op_n_arg(expr); i++) {
    isl_ast_expr* arg = isl_ast_expr_get_op_arg(expr, i);
    Value value = buildIslExpr(arg, ivToValue_, builder_);
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

FailureOr<scf::ValueVector> MLIRLoopNestGenerator::visitAstNode(
    isl_ast_node* node, BodyBuilderFn bodyBuilder) {
  switch (isl_ast_node_get_type(node)) {
    case isl_ast_node_for:
      return visitAstNodeFor(node, bodyBuilder);
    case isl_ast_node_if:
      return visitAstNodeIf(node, bodyBuilder);
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
    BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder_);
  isl_ast_node* tree = constructAst(rel, ctx_);
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
