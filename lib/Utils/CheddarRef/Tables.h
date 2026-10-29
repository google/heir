#ifndef LIB_UTILS_CHEDDARREF_TABLES_H_
#define LIB_UTILS_CHEDDARREF_TABLES_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// Precomputed constant tables for Cheddar's GPU kernels at word width W (32 or
/// 64).
template <unsigned W>
struct Tables {
  static_assert(W == 32 || W == 64, "Tables only supports W=32 or W=64");

  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  /// Moduli in Cheddar limb order (ter reversed, main, aux). Size:
  /// total_primes.
  std::vector<Word> primes;

  /// Montgomery inverse q^-1 mod 2^W for each prime. Size: total_primes.
  std::vector<SignedWord> q_inv;

  /// R mod q = 2^W mod q for each prime. Size: total_primes.
  std::vector<Word> R_mod_q;

  /// R^2 mod q = (2^W)^2 mod q for each prime. Size: total_primes.
  std::vector<Word> R2_mod_q;

  /// N^-1 mod q for each prime. Size: total_primes.
  std::vector<Word> N_inv;

  /// (N^-1 * R) mod q for each prime. Size: total_primes.
  std::vector<Word> N_inv_mont;

  /// Bit-reversed Montgomery twiddle factors for forward NTT: psi_rev[j] * R
  /// mod q. Flattened 2D table of shape [twiddle_limbs, degree].
  size_t twiddle_limbs = 0;
  size_t degree = 0;
  std::vector<Word> psi_rev_mont;

  /// Bit-reversed Montgomery twiddle factors for inverse NTT: psi_inv_rev[j] *
  /// R mod q. Flattened 2D table of shape [twiddle_limbs, degree].
  std::vector<Word> psi_inv_rev_mont;

  /// BConv prescale table: ((prod_{j != i} q_j) * N)^-1 mod q_i.
  /// Size: num_main (bconv_src_len).
  size_t bconv_src_len = 0;
  size_t bconv_dst_len = 0;
  std::vector<Word> bconv_mod_switch1;

  /// BConv projection table: Normalize((prod_{j != i} q_j) * R^2 mod p_m, p_m).
  /// Flattened 2D table of shape [num_aux (bconv_dst_len), num_main
  /// (bconv_src_len)].
  std::vector<SignedWord> bconv_mod_switch2;

  /// Galois factors: 5^k mod 2N for k in [0, degree / 2].
  /// Size: degree / 2 + 1.
  std::vector<Word> galois_factors;

  /// Complex conjugation Galois automorphism factor: 2N - 1.
  Word galois_conj = 0;

  /// Generates all constant tables from the given parameters.
  /// `max_limbs` specifies how many limbs of twiddle factors to generate (0 =
  /// all).
  static Tables<W> Generate(const Params<W>& params, size_t max_limbs = 8);

  /// Modular and number-theoretic helper functions (pure integer, no floating
  /// point):

  /// Integer square root floor(sqrt(n)) via integer Newton's method.
  static Word IntSqrt(Word n);

  /// Modular multiplication (a * b) % q.
  static Word MultMod(Word a, Word b, Word q);

  /// Modular exponentiation (base^exp) % q.
  static Word PowMod(Word base, Word exp, Word q);

  /// Modular inverse a^-1 mod q using the extended Euclidean algorithm.
  static Word InvMod(Word a, Word q);

  /// Finds the smallest generator in the finite field GF(q), matching Cheddar's
  /// FindGenerator.
  static Word FindGenerator(Word q);

  /// Finds a primitive M-th root of unity in GF(q) matching Cheddar's
  /// FindPrimitiveMthRoot.
  static Word FindPrimitiveMthRoot(Word M, Word q);

  /// Computes ModDown epilogue inv_prime_prod: (prod(src_primes))^-1 * R mod
  /// dst_primes[i].
  static std::vector<Word> ComputeModDownEpilogue(
      const std::vector<Word>& src_primes, const std::vector<Word>& dst_primes);
};

using Tables32 = Tables<32>;
using Tables64 = Tables<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_TABLES_H_
