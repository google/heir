#ifndef LIB_UTILS_CHEDDARREF_REFKEYSWITCH_H_
#define LIB_UTILS_CHEDDARREF_REFKEYSWITCH_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "lib/Utils/CheddarRef/ModSwitchTables.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/RefKernels.h"
#include "lib/Utils/CheddarRef/RefModSwitch.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir::heir::cheddar_ref {

template <unsigned W>
struct RefEvaluationKey {
  using Word = typename MontArithTraits<W>::Word;
  int beta = 0;
  size_t num_total_primes = 0;
  size_t degree = 0;
  std::vector<std::vector<Word>> bx;
  std::vector<std::vector<Word>> ax;
};

template <unsigned W>
struct RefTestKeys {
  using Word = typename MontArithTraits<W>::Word;
  size_t num_total_primes = 0;
  size_t degree = 0;
  int beta = 0;
  std::vector<Word> secret_key;
  RefEvaluationKey<W> relin_key;
  RefEvaluationKey<W> conj_key;
  std::map<int, RefEvaluationKey<W>> rot_keys;
};

template <unsigned W>
class RefKeySwitch {
 public:
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  // Generates deterministic host test keys using identical distribution and
  // format as the device harness (NTT in eval domain, ternary s, uniform ax,
  // discrete Gaussian ex).
  static RefTestKeys<W> MakeTestKeys(const Params<W>& params,
                                     const Tables<W>& tables, uint64_t seed,
                                     const std::vector<int>& rot_indices = {1,
                                                                            2});

  // Permute polynomial buffer in bit-reversed evaluation domain.
  static void Permute(Word* dst, const Word* src, size_t num_primes,
                      size_t degree, unsigned int log_n,
                      uint32_t galois_factor);

  static void PermuteConjugate(Word* dst, const Word* src, size_t num_primes,
                               size_t degree, unsigned int log_n);

  // KeySwitch without ModDown: computes accum = mod_up(in) * evk + (p_prod *
  // in_bx, p_prod * in_ax) in shape [num_q + num_aux, degree] for accum_bx and
  // accum_ax.
  static void KeySwitchNoModDown(Word* accum_bx, Word* accum_ax, const Word* in,
                                 const Word* in_bx, const Word* in_ax,
                                 const RefEvaluationKey<W>& evk, int level,
                                 const ModUpTableGroup<W>& mod_up_tables,
                                 const Tables<W>& general_tables,
                                 const Params<W>& params);

  // KeySwitch: KeySwitchNoModDown + ModDown
  static void KeySwitch(Word* out_bx, Word* out_ax, const Word* in_bx,
                        const Word* in_ax, const RefEvaluationKey<W>& evk,
                        int level, const ModUpTableGroup<W>& mod_up_tables,
                        const ModDownTableGroup<W>& mod_down_tables,
                        const Tables<W>& general_tables,
                        const Params<W>& params);

  // Relinearize: KeySwitch on in_rx + p_prod * (in_bx, in_ax) + ModDown
  static void Relinearize(Word* out_bx, Word* out_ax, const Word* in_bx,
                          const Word* in_ax, const Word* in_rx,
                          const RefEvaluationKey<W>& relin_key, int level,
                          const ModUpTableGroup<W>& mod_up_tables,
                          const ModDownTableGroup<W>& mod_down_tables,
                          const Tables<W>& general_tables,
                          const Params<W>& params);

  // RelinearizeRescale: KeySwitch on in_rx + p_prod * (in_bx, in_ax) +
  // ModDownAndRescale
  static void RelinearizeRescale(
      Word* out_bx, Word* out_ax, const Word* in_bx, const Word* in_ax,
      const Word* in_rx, const RefEvaluationKey<W>& relin_key, int level,
      const ModUpTableGroup<W>& mod_up_tables,
      const ModDownRescaleTableGroup<W>& mod_down_rescale_tables,
      const Tables<W>& general_tables, const Params<W>& params);

  // HRot: KeySwitch + Permute
  static void HRot(Word* out_bx, Word* out_ax, const Word* in_bx,
                   const Word* in_ax, const RefEvaluationKey<W>& rot_key,
                   int level, int rot_dist,
                   const ModUpTableGroup<W>& mod_up_tables,
                   const ModDownTableGroup<W>& mod_down_tables,
                   const Tables<W>& general_tables, const Params<W>& params);

  // HConj: KeySwitch + PermuteConjugate
  static void HConj(Word* out_bx, Word* out_ax, const Word* in_bx,
                    const Word* in_ax, const RefEvaluationKey<W>& conj_key,
                    int level, const ModUpTableGroup<W>& mod_up_tables,
                    const ModDownTableGroup<W>& mod_down_tables,
                    const Tables<W>& general_tables, const Params<W>& params);

  // HRotAdd: HRot(in1) + in2
  static void HRotAdd(Word* out_bx, Word* out_ax, const Word* in1_bx,
                      const Word* in1_ax, const Word* in2_bx,
                      const Word* in2_ax, const RefEvaluationKey<W>& rot_key,
                      int level, int rot_dist,
                      const ModUpTableGroup<W>& mod_up_tables,
                      const ModDownTableGroup<W>& mod_down_tables,
                      const Tables<W>& general_tables, const Params<W>& params);

  // Encrypt on host for test messages: bx = -ax * sx + mx + ex (eval form)
  static void Encrypt(Word* bx, Word* ax, const Word* mx, const Word* sx,
                      size_t num_q, size_t prime_offset, size_t degree,
                      const Params<W>& params, const Tables<W>& general_tables,
                      uint64_t seed);

  // Decrypt on host: mx = ax * sx + bx (eval form)
  static void Decrypt(Word* mx, const Word* bx, const Word* ax, const Word* sx,
                      size_t num_q, size_t prime_offset, size_t degree,
                      const Params<W>& params, const Tables<W>& general_tables);
};

using RefKeySwitch32 = RefKeySwitch<32>;
using RefKeySwitch64 = RefKeySwitch<64>;

}  // namespace mlir::heir::cheddar_ref

#endif  // LIB_UTILS_CHEDDARREF_REFKEYSWITCH_H_
