#include "lib/Utils/CheddarRef/Tables.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_ref {

namespace {

template <typename T>
T ExtendedGcd(T a, T b, T& x, T& y) {
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  T x1 = 0, y1 = 0;
  T gcd = ExtendedGcd(b % a, a, x1, y1);
  x = y1 - (b / a) * x1;
  y = x1;
  return gcd;
}

}  // namespace

template <unsigned W>
typename Tables<W>::Word Tables<W>::IntSqrt(Word n) {
  if (n <= 1) return n;
  Word x0 = n / 2;
  Word x1 = (x0 + n / x0) / 2;
  while (x1 < x0) {
    x0 = x1;
    x1 = (x0 + n / x0) / 2;
  }
  return x0;
}

template <unsigned W>
typename Tables<W>::Word Tables<W>::MultMod(Word a, Word b, Word q) {
  using DoubleWord = typename MontArithTraits<W>::DoubleWord;
  return static_cast<Word>((static_cast<DoubleWord>(a) * b) % q);
}

template <unsigned W>
typename Tables<W>::Word Tables<W>::PowMod(Word base, Word exp, Word q) {
  Word res = 1;
  base = base % q;
  while (exp > 0) {
    if (exp & 1) {
      res = MultMod(res, base, q);
    }
    exp >>= 1;
    base = MultMod(base, base, q);
  }
  return res;
}

template <unsigned W>
typename Tables<W>::Word Tables<W>::InvMod(Word a, Word q) {
  a = a % q;
  assert(a != 0 && "Inverse of 0 is undefined in InvMod");
  SignedWord x = 0, y = 0;
  ExtendedGcd(static_cast<SignedWord>(a), static_cast<SignedWord>(q), x, y);
  if (x < 0) x += static_cast<SignedWord>(q);
  return static_cast<Word>(x);
}

template <unsigned W>
typename Tables<W>::Word Tables<W>::FindGenerator(Word q) {
  Word phi = q - 1;
  ::llvm::APInt phi_ap(64, static_cast<uint64_t>(phi));
  std::vector<::llvm::APInt> factors_ap = factorize(phi_ap);
  std::vector<Word> prime_factors;
  prime_factors.reserve(factors_ap.size());
  for (const ::llvm::APInt& factor : factors_ap) {
    prime_factors.push_back(static_cast<Word>(factor.getZExtValue()));
  }

  for (Word r = 2; r <= phi; ++r) {
    bool found = true;
    for (Word factor : prime_factors) {
      if (PowMod(r, phi / factor, q) == 1) {
        found = false;
        break;
      }
    }
    if (found) return r;
  }
  assert(false && "Cannot find a generator");
  return 1;
}

template <unsigned W>
typename Tables<W>::Word Tables<W>::FindPrimitiveMthRoot(Word M, Word q) {
  assert((q - 1) % M == 0 && "Prime number is not NTT-friendly");
  Word g = FindGenerator(q);
  Word factor = (q - 1) / M;
  return PowMod(g, factor, q);
}

template <unsigned W>
std::vector<typename Tables<W>::Word> Tables<W>::ComputeModDownEpilogue(
    const std::vector<Word>& src_primes, const std::vector<Word>& dst_primes) {
  std::vector<Word> inv_p_prod(dst_primes.size(), 1);
  for (size_t i = 0; i < dst_primes.size(); ++i) {
    Word mod_prime = dst_primes[i];
    for (Word src_p : src_primes) {
      inv_p_prod[i] = MultMod(inv_p_prod[i], src_p, mod_prime);
    }
    inv_p_prod[i] = InvMod(inv_p_prod[i], mod_prime);
    inv_p_prod[i] = MontArith<W>::ToMont(inv_p_prod[i], mod_prime);
  }
  return inv_p_prod;
}

template <unsigned W>
Tables<W> Tables<W>::Generate(const Params<W>& params, size_t max_limbs) {
  Tables<W> t;
  t.primes = params.all_primes;
  size_t total_primes = t.primes.size();
  t.degree = params.degree;
  t.twiddle_limbs =
      (max_limbs > 0) ? std::min(max_limbs, total_primes) : total_primes;

  // 1. q_inv
  t.q_inv.resize(total_primes);
  for (size_t i = 0; i < total_primes; ++i) {
    t.q_inv[i] = MontArith<W>::ComputeQInv(t.primes[i]);
  }

  // 2. R_mod_q
  t.R_mod_q.resize(total_primes);
  for (size_t i = 0; i < total_primes; ++i) {
    t.R_mod_q[i] = MontArith<W>::ComputeR(t.primes[i]);
  }

  // 3. R2_mod_q
  t.R2_mod_q.resize(total_primes);
  for (size_t i = 0; i < total_primes; ++i) {
    t.R2_mod_q[i] = MontArith<W>::ComputeR2(t.primes[i]);
  }

  // 4. N_inv
  t.N_inv.resize(total_primes);
  Word deg = static_cast<Word>(t.degree);
  for (size_t i = 0; i < total_primes; ++i) {
    t.N_inv[i] = InvMod(deg, t.primes[i]);
  }

  // 5. N_inv_mont
  t.N_inv_mont.resize(total_primes);
  for (size_t i = 0; i < total_primes; ++i) {
    t.N_inv_mont[i] = MontArith<W>::ToMont(t.N_inv[i], t.primes[i]);
  }

  // 6. Twiddles: psi_rev_mont, psi_inv_rev_mont
  t.psi_rev_mont.resize(t.twiddle_limbs * t.degree);
  t.psi_inv_rev_mont.resize(t.twiddle_limbs * t.degree);

  for (size_t limb = 0; limb < t.twiddle_limbs; ++limb) {
    Word p = t.primes[limb];
    Word psi = FindPrimitiveMthRoot(2 * deg, p);
    Word psi_inv = InvMod(psi, p);

    std::vector<Word> psi_rev(t.degree);
    std::vector<Word> psi_inv_rev(t.degree);
    psi_rev[0] = 1;
    psi_inv_rev[0] = 1;
    for (size_t j = 1; j < t.degree; ++j) {
      psi_rev[j] = MultMod(psi_rev[j - 1], psi, p);
      psi_inv_rev[j] = MultMod(psi_inv_rev[j - 1], psi_inv, p);
    }

    int log_slots = params.log_n;
    for (size_t i = 0; i < t.degree; ++i) {
      size_t j = MontArith<W>::BitReverse(i, log_slots);
      if (i < j) {
        std::swap(psi_rev[i], psi_rev[j]);
        std::swap(psi_inv_rev[i], psi_inv_rev[j]);
      }
    }

    for (size_t j = 0; j < t.degree; ++j) {
      t.psi_rev_mont[limb * t.degree + j] = MontArith<W>::ToMont(psi_rev[j], p);
      t.psi_inv_rev_mont[limb * t.degree + j] =
          MontArith<W>::ToMont(psi_inv_rev[j], p);
    }
  }

  // 7. BConv tables
  const auto& src_primes = params.main_primes;
  const auto& dst_primes = params.aux_primes;
  t.bconv_src_len = src_primes.size();
  t.bconv_dst_len = dst_primes.size();

  t.bconv_mod_switch1.assign(t.bconv_src_len, 1);
  t.bconv_mod_switch2.assign(t.bconv_dst_len * t.bconv_src_len, 1);

  for (size_t i = 0; i < t.bconv_src_len; ++i) {
    Word modulus = src_primes[i];
    for (size_t j = 0; j < t.bconv_src_len; ++j) {
      if (j == i) continue;
      t.bconv_mod_switch1[i] =
          MultMod(t.bconv_mod_switch1[i], src_primes[j], modulus);
    }
    t.bconv_mod_switch1[i] = MultMod(t.bconv_mod_switch1[i], deg, modulus);
    t.bconv_mod_switch1[i] = InvMod(t.bconv_mod_switch1[i], modulus);
  }

  for (size_t m = 0; m < t.bconv_dst_len; ++m) {
    Word mod_prime = dst_primes[m];
    for (size_t i = 0; i < t.bconv_src_len; ++i) {
      Word accum = 1;
      for (size_t j = 0; j < t.bconv_src_len; ++j) {
        if (i == j) continue;
        accum = MultMod(accum, src_primes[j], mod_prime);
      }
      accum = MontArith<W>::ToMont(accum, mod_prime);
      accum = MontArith<W>::ToMont(accum, mod_prime);
      t.bconv_mod_switch2[m * t.bconv_src_len + i] =
          MontArith<W>::Normalize(accum, mod_prime);
    }
  }

  // 8. Galois factors: (5^k) % (2 * degree)
  size_t max_num_slots = t.degree / 2;
  Word m = 2 * deg;
  t.galois_factors.resize(max_num_slots + 1);
  t.galois_factors[max_num_slots] = 1;
  t.galois_factors[0] = 1;
  for (size_t i = 1; i < max_num_slots; ++i) {
    t.galois_factors[i] =
        static_cast<Word>((static_cast<typename MontArithTraits<W>::DoubleWord>(
                               t.galois_factors[i - 1]) *
                           5) %
                          m);
  }
  t.galois_conj = m - 1;

  return t;
}

template struct Tables<32>;
template struct Tables<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
