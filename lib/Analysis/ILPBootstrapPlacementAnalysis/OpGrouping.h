#ifndef LIB_ANALYSIS_ILPBOOTSTRAPPLACEMENTANALYSIS_OPGROUPING_H_
#define LIB_ANALYSIS_ILPBOOTSTRAPPLACEMENTANALYSIS_OPGROUPING_H_

#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

bool isMultiplication(Operation* op);
bool isAdditionLike(Operation* op);
bool isConstantLike(Value value);
bool shouldTrackOperation(Operation& op, DataFlowSolver* solver);

// One ILP decision class: a set of ops that share one set of decision
// variables (input level/scale, rescale count, bootstrap) and one output
// state. Grouping ops shrinks the ILP without changing its optimum.
struct OpGroup {
  // All member ops, in program order.
  SmallVector<Operation*, 4> members;
  // The member whose operands and results define the group's constraint
  // structure and variable names.
  Operation* representative = nullptr;
  // Values that carry the group's output state: the result of each merged
  // management site. Management chosen for the group is decoded after each of
  // these values.
  SmallVector<Value, 4> resultValues;
  // Addition-tree interior results. These are consumed only inside the group
  // and stay at the group's input state; no management is decoded for them.
  SmallVector<Value, 4> interiorValues;
  // Number of management sites merged into this group: bootstrap and rescale
  // decisions are charged weight times in the objective and decoded after
  // each value in resultValues.
  int weight = 1;
  bool isMultiplication = false;
  // Longest-path depth of the deepest member; group order in OpGrouping is
  // topological by this depth.
  int depth = 0;
};

struct OpGrouping {
  // Topologically ordered by (depth, program order).
  SmallVector<OpGroup> groups;
  // Tracked op -> index into groups.
  DenseMap<Operation*, int> groupIdOf;
  // Maps a group-output value to the representative value whose ILP variables
  // it shares. Values absent from the map are their own representative.
  DenseMap<Value, Value> valueRep;

  Value canonicalValue(Value value) const {
    auto it = valueRep.find(value);
    return it == valueRep.end() ? value : it->second;
  }
};

// Group the tracked operations of a secret.generic body. With compress false,
// every op is its own group.
OpGrouping computeOpGrouping(Block* body, DataFlowSolver* solver,
                             bool compress);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ILPBOOTSTRAPPLACEMENTANALYSIS_OPGROUPING_H_
