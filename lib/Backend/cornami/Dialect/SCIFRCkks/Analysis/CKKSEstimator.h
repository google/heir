#ifndef CORNAMI_CKKSESTIMATOR_PASS_H_
#define CORNAMI_CKKSESTIMATOR_PASS_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/DialectResourceMap.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

namespace heir {

enum CKKSOpcode {
  ADD,
  ADDPLAIN,
  SUB,
  SUBPLAIN,
  MUL,
  MULPLAIN,
  ROTATE,
  EXTRACT,
  NEGATE,
  RELINEARIZE,
  RESCALE,
  UNKNOWN_CKKS
};

struct CKKSLongestChain {
  std::vector<int> node_indexes;
  std::vector<std::string> node_names;
  std::vector<CKKSOpcode> opcodes;

  void reset();
  void print() const;
};

struct CKKSOpsData {
  static const int BATCH_SIZE = 4;
  std::vector<int> core_counts;
  std::vector<int> ops_count;
  int n_accumulated_throughput = 0;
  std::unordered_map<std::string, CKKSOpcode> nodes;
  std::unordered_map<std::string, std::unordered_set<std::string>> adjacencies;
  CKKSLongestChain longest_chain;

  CKKSOpsData() {
    core_counts.resize(CKKSOpcode::UNKNOWN_CKKS, 0);
    ops_count.resize(CKKSOpcode::UNKNOWN_CKKS, 0);
  }

  cornami::DialectResourceMap get_resource_map() const;
  void reset();
  void display(const std::vector<CKKSOpcode>& opcodes) const;
  void display() const;

  static std::string opcode_name(CKKSOpcode opcode);
  static std::vector<CKKSOpcode> opcodes();
  static int get_core_counts(CKKSOpcode opcode);
  static int get_throughput(CKKSOpcode opcode);
  static int get_latency(CKKSOpcode opcode);
  static std::vector<int> get_core_count_and_throughput(CKKSOpcode opcode);
};

#define GEN_PASS_DECL_CKKSESTIMATOR
#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/Passes.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // CORNAMI_CKKSESTIMATOR_PASS_H_
