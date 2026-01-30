#ifndef CORNAMI_OPENFHE_ESTIMATOR_PASS_H_
#define CORNAMI_OPENFHE_ESTIMATOR_PASS_H_

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/DialectResourceMap.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

namespace heir {

enum OpenfheOpcode {
  AddOp,
  AutomorphOp,
  DecryptOp,
  EncryptOp,
  GenContextOp,
  GenMulKeyOp,
  GenParamsOp,
  GenRotKeyOp,
  KeySwitchOp,
  LevelReduceOp,
  ModReduceOp,
  MulConstOp,
  MulNoRelinOp,
  MulOp,
  MulPlainOp,
  NegateOp,
  RelinOp,
  RotOp,
  SquareOp,
  SubOp,
  UNKNOWN_OPENFHE
};

struct OpenfheLongestChain {
  std::vector<int> node_indexes;
  std::vector<std::string> node_names;
  std::vector<OpenfheOpcode> opcodes;
  void reset();
  void print() const;
};

struct OpenfheOpsData {
  static const int BATCH_SIZE = 4;
  std::vector<int> core_counts;
  std::vector<int> ops_count;
  int n_accumulated_throughput = 0;
  std::unordered_map<std::string, OpenfheOpcode> nodes;
  std::unordered_map<std::string, std::unordered_set<std::string>> adjacencies;
  OpenfheLongestChain longest_chain;

  OpenfheOpsData() {
    core_counts.resize(OpenfheOpcode::UNKNOWN_OPENFHE, 0);
    ops_count.resize(OpenfheOpcode::UNKNOWN_OPENFHE, 0);
  }

  cornami::DialectResourceMap get_resource_map() const;
  void reset();
  void display(const std::vector<OpenfheOpcode>& opcodes) const;
  void display() const;

  static std::string opcode_name(OpenfheOpcode opcode);
  static std::vector<OpenfheOpcode> opcodes();
  static int get_core_counts(OpenfheOpcode opcode);
  static int get_throughput(OpenfheOpcode opcode);
  static int get_latency(OpenfheOpcode opcode);
  static std::vector<int> get_core_count_and_throughput(OpenfheOpcode opcode);
};

#define GEN_PASS_DECL_OPENFHEESTIMATOR
#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/Passes.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // CORNAMI_OPENFHE_ESTIMATOR_PASS_H_
