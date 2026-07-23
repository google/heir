#include "lib/Analysis/ILPBootstrapPlacementAnalysis/OpGrouping.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool isMultiplication(Operation* op) {
  return isa<arith::MulFOp, arith::MulIOp>(op);
}

bool isAdditionLike(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op);
}

bool isConstantLike(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() != nullptr;
}

static bool hasSecretResult(Operation& op, DataFlowSolver* solver) {
  return llvm::any_of(op.getResults(), [&](OpResult result) {
    return isSecret(result, solver);
  });
}

bool shouldTrackOperation(Operation& op, DataFlowSolver* solver) {
  return !isa<secret::YieldOp>(op) && hasSecretResult(op, solver);
}

namespace {

struct GroupingContext {
  GroupingContext(Block* body, DataFlowSolver* solver)
      : body(body), solver(solver) {}

  Block* body;
  DataFlowSolver* solver;
  SmallVector<Operation*> trackedOps;
  // Program-order index and longest-path depth of each tracked op.
  DenseMap<Operation*, int> opIndex;
  DenseMap<Operation*, int> opDepth;
};

}  // namespace

static OpGroup makeSingletonGroup(GroupingContext& ctx, Operation* op) {
  OpGroup group;
  group.members.push_back(op);
  group.representative = op;
  for (OpResult result : op->getResults()) {
    if (isSecret(result, ctx.solver)) group.resultValues.push_back(result);
  }
  group.weight = 1;
  group.isMultiplication = isMultiplication(op);
  group.depth = ctx.opDepth.lookup(op);
  return group;
}

// Collapse maximal addition trees into one group each (Orbit's
// addition_squash). All additions in a tree must execute at the same level
// and scale, so they need only one set of ILP variables and one management
// decision after the final (deepest) addition. An addition joins the tree of
// a consumer only when its entire fanout is inside that tree, so no interior
// sum escapes at a possibly different state.
static SmallVector<OpGroup> squashAdditionTrees(GroupingContext& ctx,
                                                bool compress) {
  auto isSquashable = [&](Operation* op) {
    return compress && isAdditionLike(op) && op->getNumResults() == 1;
  };

  // Grow trees from their deepest member so producers join consumers.
  SmallVector<Operation*> order(ctx.trackedOps);
  llvm::stable_sort(order, [&](Operation* a, Operation* b) {
    return ctx.opDepth.lookup(a) > ctx.opDepth.lookup(b);
  });

  SmallVector<OpGroup> groups;
  DenseSet<Operation*> used;
  for (Operation* head : order) {
    if (used.contains(head)) continue;
    if (!isSquashable(head)) {
      used.insert(head);
      groups.push_back(makeSingletonGroup(ctx, head));
      continue;
    }

    // Fixpoint: absorb an addition when all users of its result are already
    // in the tree. Absorbing one addition can make its producers eligible.
    DenseSet<Operation*> tree;
    tree.insert(head);
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation*> candidates;
      for (Operation* member : tree) {
        for (Value operand : member->getOperands()) {
          Operation* def = operand.getDefiningOp();
          if (!def || def->getBlock() != ctx.body) continue;
          if (used.contains(def) || tree.contains(def)) continue;
          if (!ctx.opIndex.contains(def) || !isSquashable(def)) continue;
          candidates.push_back(def);
        }
      }
      for (Operation* candidate : candidates) {
        if (tree.contains(candidate)) continue;
        bool hasUser = false;
        bool allUsersInside = true;
        for (Operation* user : candidate->getResult(0).getUsers()) {
          hasUser = true;
          if (!tree.contains(user)) {
            allUsersInside = false;
            break;
          }
        }
        if (hasUser && allUsersInside) {
          tree.insert(candidate);
          changed = true;
        }
      }
    }

    OpGroup group;
    group.members.append(tree.begin(), tree.end());
    llvm::stable_sort(group.members, [&](Operation* a, Operation* b) {
      return ctx.opIndex.lookup(a) < ctx.opIndex.lookup(b);
    });
    group.representative = head;
    group.resultValues.push_back(head->getResult(0));
    for (Operation* member : group.members) {
      used.insert(member);
      if (member != head) group.interiorValues.push_back(member->getResult(0));
    }
    group.weight = 1;
    group.isMultiplication = false;
    group.depth = ctx.opDepth.lookup(head);
    groups.push_back(std::move(group));
  }
  return groups;
}

// Merge structurally equivalent groups by compression (Orbit's auto_compress):
// iterative label refinement, where each round relabels a group by its op
// class, depth, and the current labels of its producers and consumers until
// the labeling reaches a fixpoint. Two groups may merge only when they have
// the same depth, the same op class, and — at the fixpoint — the same set of
// producer classes; that guarantees replaying one representative's solution on
// every member satisfies each member's constraints, and cost linearity makes
// the merged objective exactly the sum over members.
static void mergeEquivalentGroups(GroupingContext& ctx,
                                  SmallVector<OpGroup>& groups) {
  int numGroups = groups.size();
  if (numGroups <= 1) return;

  DenseMap<Operation*, int> groupOf;
  for (auto [i, group] : llvm::enumerate(groups)) {
    for (Operation* member : group.members) groupOf[member] = i;
  }

  // Groups whose results are yielded (or that have multiple results) never
  // merge: yielded values can carry per-result mgmt.mgmt pins that must not
  // propagate to other ops.
  std::vector<int> uniqueTag(numGroups, 0);
  int nextUniqueTag = 1;
  for (int i = 0; i < numGroups; ++i) {
    bool unique = groups[i].representative->getNumResults() != 1;
    for (Value result : groups[i].resultValues) {
      for (Operation* user : result.getUsers()) {
        if (isa<secret::YieldOp>(user)) unique = true;
      }
    }
    if (unique) uniqueTag[i] = nextUniqueTag++;
  }

  // Operand labels are (kind, id) pairs so distinct kinds cannot collide:
  // kind 0 = producer group (id = current-round label), 1 = secret block
  // argument, 2 = constant, 3 = other non-secret value, 4 = secret value not
  // produced by a tracked op (never merged across).
  using OperandLabel = std::pair<int, int>;
  DenseMap<Value, int> externalIds;

  int addOpKey = 0;
  std::map<StringRef, int> opKeys;
  auto opKeyOf = [&](const OpGroup& group) {
    if (isAdditionLike(group.representative)) return addOpKey;
    auto [it, inserted] = opKeys.try_emplace(
        group.representative->getName().getStringRef(), opKeys.size() + 1);
    return it->second;
  };

  std::vector<int> order(numGroups);
  for (int i = 0; i < numGroups; ++i) order[i] = i;
  llvm::stable_sort(order, [&](int a, int b) {
    if (groups[a].depth != groups[b].depth)
      return groups[a].depth < groups[b].depth;
    return ctx.opIndex.lookup(groups[a].members.front()) <
           ctx.opIndex.lookup(groups[b].members.front());
  });

  using Descriptor =
      std::tuple<int, int, int, std::vector<OperandLabel>, std::vector<int>>;
  std::vector<int> labels(numGroups, 0);
  int numDistinct = 1;
  constexpr int kMaxRounds = 100;
  for (int round = 0; round < kMaxRounds; ++round) {
    std::vector<int> newLabels(numGroups, 0);
    std::map<Descriptor, int> intern;
    for (int i : order) {
      const OpGroup& group = groups[i];

      // Producer classes, from this round's labels (producers are strictly
      // shallower, so they are relabeled before their consumers).
      std::vector<OperandLabel> inputs;
      for (Operation* member : group.members) {
        for (Value operand : member->getOperands()) {
          Operation* def = operand.getDefiningOp();
          if (def && groupOf.count(def)) {
            if (groupOf[def] == i) continue;  // interior edge
            inputs.emplace_back(0, newLabels[groupOf[def]]);
            continue;
          }
          if (auto arg = dyn_cast<BlockArgument>(operand);
              arg && arg.getOwner() == ctx.body &&
              isSecret(operand, ctx.solver)) {
            inputs.emplace_back(1, arg.getArgNumber());
            continue;
          }
          if (isConstantLike(operand)) {
            inputs.emplace_back(2, 0);
            continue;
          }
          if (!isSecret(operand, ctx.solver)) {
            inputs.emplace_back(3, 0);
            continue;
          }
          auto [it, inserted] =
              externalIds.try_emplace(operand, externalIds.size());
          inputs.emplace_back(4, it->second);
        }
      }
      llvm::sort(inputs);
      inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());

      // Consumer classes, from the previous round's labels.
      std::vector<int> successors;
      for (Value result : group.resultValues) {
        for (Operation* user : result.getUsers()) {
          if (groupOf.count(user)) successors.push_back(labels[groupOf[user]]);
        }
      }
      llvm::sort(successors);
      successors.erase(std::unique(successors.begin(), successors.end()),
                       successors.end());

      Descriptor descriptor(group.depth, opKeyOf(group), uniqueTag[i],
                            std::move(inputs), std::move(successors));
      auto [it, inserted] =
          intern.try_emplace(std::move(descriptor), intern.size());
      newLabels[i] = it->second;
    }
    int newDistinct = intern.size();
    labels = std::move(newLabels);
    if (newDistinct == numDistinct) break;
    numDistinct = newDistinct;
  }

  if (numDistinct == numGroups) return;  // nothing merged

  std::map<int, std::vector<int>> buckets;
  for (int i = 0; i < numGroups; ++i) buckets[labels[i]].push_back(i);

  SmallVector<OpGroup> merged;
  for (auto& [label, bucket] : buckets) {
    OpGroup group = std::move(groups[bucket.front()]);
    for (size_t k = 1; k < bucket.size(); ++k) {
      OpGroup& other = groups[bucket[k]];
      group.members.append(other.members.begin(), other.members.end());
      group.resultValues.append(other.resultValues.begin(),
                                other.resultValues.end());
      group.interiorValues.append(other.interiorValues.begin(),
                                  other.interiorValues.end());
      group.weight += other.weight;
    }
    llvm::stable_sort(group.members, [&](Operation* a, Operation* b) {
      return ctx.opIndex.lookup(a) < ctx.opIndex.lookup(b);
    });
    merged.push_back(std::move(group));
  }
  groups = std::move(merged);
}

OpGrouping computeOpGrouping(Block* body, DataFlowSolver* solver,
                             bool compress) {
  GroupingContext ctx(body, solver);
  int index = 0;
  for (Operation& op : body->getOperations()) {
    if (!shouldTrackOperation(op, solver)) continue;
    ctx.trackedOps.push_back(&op);
    ctx.opIndex[&op] = index++;
    int depth = 0;
    for (Value operand : op.getOperands()) {
      Operation* def = operand.getDefiningOp();
      auto it = ctx.opDepth.find(def);
      if (def && it != ctx.opDepth.end())
        depth = std::max(depth, it->second + 1);
    }
    ctx.opDepth[&op] = depth;
  }

  SmallVector<OpGroup> groups = squashAdditionTrees(ctx, compress);
  if (compress) mergeEquivalentGroups(ctx, groups);

  llvm::stable_sort(groups, [&](const OpGroup& a, const OpGroup& b) {
    if (a.depth != b.depth) return a.depth < b.depth;
    return ctx.opIndex.lookup(a.members.front()) <
           ctx.opIndex.lookup(b.members.front());
  });

  OpGrouping grouping;
  grouping.groups = std::move(groups);
  for (auto [i, group] : llvm::enumerate(grouping.groups)) {
    for (Operation* member : group.members) grouping.groupIdOf[member] = i;
    if (group.representative->getNumResults() == 1) {
      Value repResult = group.representative->getResult(0);
      for (Value result : group.resultValues) {
        if (result != repResult) grouping.valueRep[result] = repResult;
      }
    }
  }
  return grouping;
}

}  // namespace heir
}  // namespace mlir
