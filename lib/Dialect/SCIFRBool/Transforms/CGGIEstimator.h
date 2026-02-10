#ifndef CORNAMI_CGGIESTIMATOR_PASS_H_
#define CORNAMI_CGGIESTIMATOR_PASS_H_

#include <array>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/SCIFRBool/Transforms/DialectResourceMap.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

namespace heir {

struct LWEParameters {
  uint64_t LWE_N;  // polynomial degree
  uint64_t LWE_l;  // levels of ciphertext in CGGI
  uint64_t LWE_k;  // GLWE parameter
  uint64_t LWE_n;  // LWE dimension

  explicit LWEParameters(uint64_t polynomial_degree, uint64_t levels,
                         uint64_t glwe_param, uint64_t lwe_dim)
      : LWE_N(polynomial_degree),
        LWE_l(levels),
        LWE_k(glwe_param),
        LWE_n(lwe_dim) {}
};
struct CGGIOpsData : LWEParameters {
  explicit CGGIOpsData(uint64_t polynomial_degree, uint64_t levels,
                       uint64_t glwe_param, uint64_t lwe_dim)
      : LWEParameters(polynomial_degree, levels, glwe_param, lwe_dim){};

  enum CGGIOpsEnum {
    AND,
    LUT2,
    LUT3,
    LUTLINCOMB,
    MULTILUTLINCOMB,
    NAND,
    NOR,
    NOT,
    OR,
    PACKED,
    XNOR,
    XOR,
    UNKNOWN
  };

  CGGIOpsEnum get_op_kind(mlir::Operation* op);

  CGGIOpsEnum to_enum(std::string opname) const {
    if (opname == "AND") return AND;
    if (opname == "LUT2") return LUT2;
    if (opname == "LUT3") return LUT3;
    if (opname == "LUTLINCOMB") return LUTLINCOMB;
    if (opname == "MULTILUTLINCOMB") return MULTILUTLINCOMB;
    if (opname == "NAND") return NAND;
    if (opname == "NOR") return NOR;
    if (opname == "NOT") return NOT;
    if (opname == "OR") return OR;
    if (opname == "PACKED") return PACKED;
    if (opname == "XNOR") return XNOR;
    if (opname == "XOR") return XOR;
    return UNKNOWN;
  }

  using DialectBootstrapAndKeySwitchMap =
      std::map<CGGIOpsEnum, std::array<bool, 2>>;  // does operator need
                                                   // bootstrap and keyswitch ?

  // std::array[3] = count, ncores, ncycles
  std::array<uint64_t, 3> n_andop = {
      0,
  };  //::mlir::heir::cggi::AndOp;
  std::array<uint64_t, 3> n_lut2op = {
      0,
  };  //::mlir::heir::cggi::Lut2Op;
  std::array<uint64_t, 3> n_lut3op = {
      0,
  };  //::mlir::heir::cggi::Lut3Op,
  std::array<uint64_t, 3> n_lutlincombop = {
      0,
  };  //::mlir::heir::cggi::LutLinCombOp,
  std::array<uint64_t, 3> n_multilutlincombop = {
      0,
  };  //::mlir::heir::cggi::MultiLutLinCombOp,
  std::array<uint64_t, 3> n_nandop = {
      0,
  };  //::mlir::heir::cggi::NandOp,
  std::array<uint64_t, 3> n_norop = {
      0,
  };  //::mlir::heir::cggi::NorOp,
  std::array<uint64_t, 3> n_notop = {
      0,
  };  //::mlir::heir::cggi::NotOp,
  std::array<uint64_t, 3> n_orop = {
      0,
  };  //::mlir::heir::cggi::OrOp,
  std::array<uint64_t, 3> n_packedop = {
      0,
  };  //::mlir::heir::cggi::PackedOp,
  std::array<uint64_t, 3> n_xnorop = {
      0,
  };  //::mlir::heir::cggi::XNorOp,
  std::array<uint64_t, 3> n_xorop = {
      0,
  };  //::mlir::heir::cggi::XorOp

  const DialectBootstrapAndKeySwitchMap m_bs_and_ks = {
      {AND, {true, true}},        {NAND, {true, true}},
      {NOR, {true, true}},        {OR, {true, true}},
      {XOR, {true, true}},        {XNOR, {true, true}},
      {NOT, {false, false}},      {PACKED, {true, true}},
      {LUT2, {true, true}},       {LUT3, {true, true}},
      {LUTLINCOMB, {true, true}}, {MULTILUTLINCOMB, {true, true}},
  };

  uint64_t get_bootstrap_resource_fc() const;
  uint64_t get_bootstrap_resource_throughput() const;
  uint64_t get_keyswitch_resource_fc() const;
  uint64_t get_keyswitch_resource_throughput() const;

  cornami::DialectResourceMap get_resource_map() const;

  cornami::OpCoreCountAndLatencyT get_resource_use(
      CGGIOpsEnum opkind, uint64_t count) const; /* get FC core */

  // number of clock cycles to get the ciphertext output
  uint64_t get_canonical_throughput() const {
    return 2 * LWE_N * LWE_n;  // clocks/sec
  }

  uint64_t get_canonical_latency(uint64_t batchsize) const {
    return batchsize * get_canonical_throughput();
  }

  bool analyze(mlir::Operation* op);
  void reset();
  void display() const;
};

#define GEN_PASS_DECL_CGGIESTIMATOR
#include "lib/Dialect/SCIFRBool/Transforms/Passes.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // CORNAMI_CGGIESTIMATOR_PASS_H_
