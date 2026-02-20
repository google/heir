#pragma once
#include <utility>
#include <vector>

namespace mlir {
namespace cornami {

enum OpType { READ, WRITE, AUTO_READ, KICK_OUT };

struct ArchCounter {
  int cyc = 0;
  int add_cyc = 0;
  int mult_cyc = 0;

  // ntt cycles for all of bootstrapping
  // otf cycles for all twiddle factors
  int ntt_cyc = 0;
  int ntt_otf = 0;

  // basis conversion
  int bsc_cyc = 0;

  // worst-case and best-case automorphism
  int auto_cyc_wc = 0;
  int auto_cyc_bc = 0;

  /**
  """
  Reads are reads in from external memory
  Writes are writes out to external memory
  Untagged dram tracks local storage needed in external memory

  The default model is that nothing is stored on chip.
  """
  */

  std::vector<std::pair<OpType, int>> op_list;
  int min_bytes = 0;

  // # dram_limb_rd: int = 0
  int _dram_limb_rd = 0;

  // # dram_limb_wr: int = 0
  int _dram_limb_wr = 0;

  // number of read and write ports used during program
  int rd_ports = 0;
  int wr_ports = 0;

  int dram_limb = 0;
  int dram_ntt_rd = 0;
  int dram_ntt = 0;

  int _dram_auto_rd = 0;

  int dram_limb_rd() const { return _dram_limb_rd; }

  void set_dram_limb_rd(int value) {
    op_list.emplace_back(
        std::make_pair(OpType::READ, (int)(value - _dram_limb_rd)));
    _dram_limb_rd = value;
  }

  int dram_limb_wr() const { return _dram_limb_wr; }

  void set_dram_limb_wr(int value) {
    op_list.emplace_back(
        std::make_pair(OpType::WRITE, (int)(value - _dram_limb_wr)));
    _dram_limb_wr = value;
  }

  int dram_auto_rd() const { return _dram_auto_rd; }

  void set_dram_auto_rd(int value) {
    op_list.emplace_back(
        std::make_pair(OpType::AUTO_READ, (int)(value - _dram_auto_rd)));
    _dram_auto_rd = value;
  }

  int dram_total_rdwr_small() {
    return (dram_limb_rd() + dram_limb_wr() + dram_auto_rd());
  }

  int total_cycle_bc() { return add_cyc + mult_cyc + auto_cyc_bc; }

  int total_cycle_wc() { return add_cyc + mult_cyc + auto_cyc_wc; }

  ArchCounter& operator+=(const ArchCounter& other) {
    _dram_limb_rd = dram_limb_rd() + other.dram_limb_rd(),
    _dram_limb_wr = dram_limb_wr() + other.dram_limb_wr(),
    _dram_auto_rd = dram_auto_rd() + other.dram_auto_rd(),
    // op_list=op_list + other.op_list,
        add_cyc += other.add_cyc;
    mult_cyc += other.mult_cyc;
    return *this;
  }
};

struct SWCounter {
  int mult = 0;
  int add = 0;
  int ntt = 0;
  SWCounter& operator+=(const SWCounter& that);
};

struct PerfCounter {
  SWCounter sw;
  ArchCounter arch_emu, arch_asic;

  PerfCounter& operator+=(const PerfCounter& that);
};

}  // namespace cornami
}  // namespace mlir
