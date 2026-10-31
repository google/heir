#ifndef LIB_UTILS_CHEDDARREF_MODSWITCHTABLES_H_
#define LIB_UTILS_CHEDDARREF_MODSWITCHTABLES_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// Tables for PseudoModUp at a given level.
/// PseudoModUp computes: out = src * p_prod mod q.
template <unsigned W>
struct PseudoModUpTableGroup {
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;

  /// Per-q-limb constant (p_prod * R) mod q_i.
  /// Size: num_q_primes.
  std::vector<Word> p_prod;
};

/// Tables for ModUp at a given level.
/// ModUp decomposes into:
///   INTT(mult_const_normalize) with mod_up1
///   Memcpy for verbatim limbs
///   BConv for remaining destination limbs with mod_up2[digit]
///   NTT(skip_range)
template <unsigned W>
struct ModUpTableGroup {
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  int level = 0;
  int num_q_primes = 0;
  int num_aux = 0;
  int beta = 0;

  /// INTT normalization constants for input Q primes:
  /// mod_up1[i] = ((prod_{j != i} q_j) * N)^-1 mod q_i.
  /// Size: num_q_primes.
  std::vector<Word> mod_up1;

  /// Per-digit BConv projection tables.
  /// Each table has shape [dst_len_i, src_len_i] where:
  ///   src_len_i = min((i+1)*num_aux, padded_q) - i*num_aux (after offset)
  ///   dst_len_i = num_q_primes - src_len_i + num_aux.
  /// Each digit table is stored row-major.
  struct DigitTable {
    int prime_index_start = 0;
    int prime_index_end = 0;
    int src_len = 0;
    int dst_len = 0;
    std::vector<SignedWord> table;  // [dst_len, src_len]
  };
  std::vector<DigitTable> digits;
};

/// Tables for ModDown at a given level.
template <unsigned W>
struct ModDownTableGroup {
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  int level = 0;
  int num_q_primes = 0;
  int num_aux = 0;

  int src_len = 0;  // num_aux
  int dst_len = 0;  // num_q_primes

  /// INTT constants for aux primes P:
  /// ((prod_{j != i} p_j) * N)^-1 mod p_i.
  /// Size: num_aux.
  std::vector<Word> mod_down1;

  /// BConv table from aux primes P to q primes:
  /// Shape [dst_len, src_len] = [num_q_primes, num_aux].
  std::vector<SignedWord> mod_down2;

  /// Epilogue inv_prime_prod: (prod(p_primes))^-1 * R mod q_i.
  /// Size: num_q_primes.
  std::vector<Word> inv_prime_prod;
};

/// Tables for Rescale at a given level.
template <unsigned W>
struct RescaleTableGroup {
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  int level = 0;
  int src_len = 0;
  int dst_len = 0;

  int pad_start = 0;
  int pad_end = 0;
  int restore_start = 0;
  int restore_end = 0;

  /// INTT constants for rescale_src_primes:
  /// Size: src_len.
  std::vector<Word> rescale1;

  /// BConv table from rescale_src_primes to rescale_dst_primes:
  /// Shape [dst_len, src_len].
  std::vector<SignedWord> rescale2;

  /// Epilogue inv_prime_prod: (prod(rescale_src_primes))^-1 * R mod
  /// dst_primes[i]. Size: dst_len.
  std::vector<Word> inv_prime_prod;

  /// Epilogue src2_padding:
  /// Size: dst_len - (restore_end - restore_start).
  std::vector<Word> src2_padding;
};

/// Tables for ModDownAndRescale at a given level.
template <unsigned W>
struct ModDownRescaleTableGroup {
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  int level = 0;
  int src_len = 0;
  int dst_len = 0;

  int pad_start = 0;
  int pad_end = 0;
  int restore_start = 0;
  int restore_end = 0;

  /// INTT constants for mod_down_rescale_src_primes:
  /// Size: src_len.
  std::vector<Word> mod_down_rescale1;

  /// BConv table from mod_down_rescale_src_primes to rescale_dst_primes:
  /// Shape [dst_len, src_len].
  std::vector<SignedWord> mod_down_rescale2;

  /// Epilogue inv_prime_prod:
  /// Size: dst_len.
  std::vector<Word> inv_prime_prod;

  /// Epilogue src2_padding:
  /// Size: dst_len - (restore_end - restore_start).
  std::vector<Word> src2_padding;
};

/// Generator functions matching Cheddar's ModSwitchHandler precomputation
/// exactly.
template <unsigned W>
class ModSwitchTables {
 public:
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  static PseudoModUpTableGroup<W> GeneratePseudoModUp(const Params<W>& params,
                                                      int level);

  static ModUpTableGroup<W> GenerateModUp(const Params<W>& params, int level);

  static ModDownTableGroup<W> GenerateModDown(const Params<W>& params,
                                              int level);

  static RescaleTableGroup<W> GenerateRescale(const Params<W>& params,
                                              int level);

  static ModDownRescaleTableGroup<W> GenerateModDownRescale(
      const Params<W>& params, int level);

  /// Low-level helpers matching Cheddar's PopulateModSwitchConstants and
  /// PopulateModDownEpilogueConstants:
  static void PopulateModSwitchConstants(std::vector<Word>& const1,
                                         std::vector<SignedWord>& const2,
                                         const std::vector<Word>& src_primes,
                                         const std::vector<Word>& dst_primes,
                                         size_t degree, int restore_start = 0,
                                         int restore_end = 0);

  static void PopulateModDownEpilogueConstants(
      std::vector<Word>& inv_prime_prod, std::vector<Word>& padding,
      const std::vector<Word>& src_primes, const std::vector<Word>& dst_primes,
      int restore_start = 0, int restore_end = 0);
};

using ModSwitchTables32 = ModSwitchTables<32>;
using ModSwitchTables64 = ModSwitchTables<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_MODSWITCHTABLES_H_
