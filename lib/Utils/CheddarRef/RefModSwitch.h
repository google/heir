#ifndef LIB_UTILS_CHEDDARREF_REFMODSWITCH_H_
#define LIB_UTILS_CHEDDARREF_REFMODSWITCH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/ModSwitchTables.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/RefBConv.h"
#include "lib/Utils/CheddarRef/RefKernels.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// Reference implementation of Cheddar's ModSwitch operations.
/// Operates on plain CPU buffers in Cheddar layout:
///   shape [limbs, degree] flattened row-major.
template <unsigned W>
class RefModSwitch {
 public:
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  /// Fused ModDown / Rescale / ModDownAndRescale epilogue on CPU.
  /// dst[l, i] = ((src2_exists ? (src2[l, i] * src2_padding[l]) : 0) -
  /// ntt_bconv[l, i]) * inv_prime_prod[l] mod q_l. where all arithmetic is in
  /// Montgomery form modulo dst_primes[l].
  static void ModDownEpilogue(Word* dst, const Word* ntt_bconv,
                              const Word* inv_prime_prod, size_t dst_len,
                              size_t degree, const Word* dst_primes,
                              const SignedWord* dst_q_inv,
                              const Word* src2 = nullptr,
                              const Word* src2_padding = nullptr,
                              size_t pad_start = 0, size_t pad_end = 0);

  /// PseudoModUp: dst = src * p_prod mod q pointwise across all limbs.
  /// src and dst are in eval form.
  static void PseudoModUp(Word* dst, const Word* src,
                          const PseudoModUpTableGroup<W>& tables,
                          const Tables<W>& general_tables,
                          const Params<W>& params, int level);

  /// ModUp: expands `src` [num_q_primes, degree] into `dsts` (beta buffers of
  /// shape [num_q_primes + num_aux, degree]).
  static void ModUp(std::vector<std::vector<Word>>& dsts, const Word* src,
                    const ModUpTableGroup<W>& tables,
                    const Tables<W>& general_tables, const Params<W>& params);

  /// ModDown: shrinks `src` [num_q_primes + num_aux, degree] down to `dst`
  /// [num_q_primes, degree].
  static void ModDown(Word* dst, const Word* src,
                      const ModDownTableGroup<W>& tables,
                      const Tables<W>& general_tables, const Params<W>& params);

  /// Rescale: shrinks `src` [num_q_primes_level, degree] down to `dst`
  /// [num_q_primes_next_level, degree].
  static void Rescale(Word* dst, const Word* src,
                      const RescaleTableGroup<W>& tables,
                      const Tables<W>& general_tables, const Params<W>& params);

  /// ModDownAndRescale: shrinks `src` [num_q_primes_level + num_aux, degree]
  /// down to `dst` [num_q_primes_next_level, degree].
  static void ModDownAndRescale(Word* dst, const Word* src,
                                const ModDownRescaleTableGroup<W>& tables,
                                const Tables<W>& general_tables,
                                const Params<W>& params);
};

using RefModSwitch32 = RefModSwitch<32>;
using RefModSwitch64 = RefModSwitch<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_REFMODSWITCH_H_
