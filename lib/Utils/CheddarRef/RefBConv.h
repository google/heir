#ifndef LIB_UTILS_CHEDDARREF_REFBCOV_H_
#define LIB_UTILS_CHEDDARREF_REFBCOV_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// Approximate basis conversion (BConv) reference implementation.
///
/// Mathematical contract:
/// - Input `src`: centered signed residues in [-(q_s-1)/2, (q_s-1)/2]
///   (coeff_centered form), typically produced by INTT with
///   mult_const_normalize epilogue.
/// - Table `bconv_table`: flattened 2D row-major matrix of shape [dst_len,
/// src_len]
///   where each entry is Normalize((prod_{j != s} q_j) * R^2 mod p_m, p_m).
/// - Output `dst`: canonical residues in [0, p_m) (coeff form).
///
/// Semantics and Proof of Value-Preserving Re-centering:
/// For each destination prime p_m and coefficient index x:
///   accum = sum_{s=0}^{src_len-1} src[s, x] * table[m, s]
/// In exact arithmetic, table[m, s] = (q̂_s * R^2) mod p_m (centered).
/// Hence:
///   accum = sum_{s=0}^{src_len-1} src[s, x] * table[m, s]
///         ≡ R^2 * sum_{s=0}^{src_len-1} src[s, x] * q̂_s  (mod p_m)
/// Applying Montgomery reduction at the end:
///   MontReduce(accum) = (accum * R^-1) mod p_m
///                     ≡ R * sum_{s=0}^{src_len-1} src[s, x] * q̂_s  (mod p_m)
///
/// Cheddar's periodic re-centering adds or subtracts prime_th = p_m * 2^W = p_m
/// * R when accum grows outside [-prime_th / 2, prime_th / 2). Because
/// (prime_th) * R^-1 = (p_m * R) * R^-1 = p_m ≡ 0 (mod p_m), adding or
/// subtracting any multiple of prime_th changes the Montgomery reduction input
/// by a multiple of p_m * R, which changes the reduction result by a multiple
/// of p_m, which is 0 mod p_m.
///
/// Therefore, periodic re-centering is VALUE-PRESERVING: it yields the exact
/// same residue mod p_m as exact accumulation without re-centering, while
/// keeping accum strictly bounded within (-p_m * R, p_m * R) to satisfy the
/// range precondition of lazy Montgomery reduction and prevent integer
/// overflow.
template <unsigned W>
class RefBConv {
 public:
  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;
  using DoubleWord = typename Traits::DoubleWord;
  using SignedDoubleWord = typename Traits::SignedDoubleWord;

  /// Performs BConv from `src` to `dst`.
  /// - `dst`: output buffer of shape [dst_len, degree] (or with skip mapping)
  /// - `src`: input buffer of shape [src_len, degree] with centered residues
  /// - `bconv_table`: shape [dst_len, src_len]
  /// - `dst_primes`: moduli p_m for destination limbs
  /// - `dst_inv_primes`: Montgomery inverses p_m^-1 mod 2^W
  /// - `skip_start`, `skip_end`: range of prime indices in dst to skip (for
  /// ModUp)
  /// - `periodic_recenter`: whether to re-center every 4 terms (as Cheddar
  /// does)
  ///   or accumulate without periodic re-centering.
  static void BConv(Word* dst, const SignedWord* src,
                    const SignedWord* bconv_table, size_t src_len,
                    size_t dst_len, size_t degree, const Word* dst_primes,
                    const SignedWord* dst_inv_primes, size_t skip_start = 0,
                    size_t skip_end = 0, bool periodic_recenter = true);
};

using RefBConv32 = RefBConv<32>;
using RefBConv64 = RefBConv<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_REFBCOV_H_
