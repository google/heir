#include "lib/Utils/CheddarRef/RefKeySwitch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "lib/Utils/CheddarRef/ModSwitchTables.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/RefKernels.h"
#include "lib/Utils/CheddarRef/RefModSwitch.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir::heir::cheddar_ref {
namespace {

constexpr double kErrorStdDev = 3.2;

class LocalSplitMix64 {
 public:
  explicit LocalSplitMix64(uint64_t seed) : state_(seed) {}
  uint64_t Next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  template <typename T>
  T NextUniform(T bound) {
    if (bound <= 1) return 0;
    return static_cast<T>(Next() % bound);
  }

 private:
  uint64_t state_;
};

template <unsigned W>
typename MontArithTraits<W>::Word ComputePProdMont(
    const Params<W>& params, typename MontArithTraits<W>::Word q) {
  using Word = typename MontArithTraits<W>::Word;
  Word p_prod = 1;
  for (Word p_aux : params.aux_primes) {
    p_prod = Tables<W>::MultMod(p_prod, p_aux, q);
  }
  return MontArith<W>::ToMont(p_prod, q);
}

}  // namespace

template <unsigned W>
void RefKeySwitch<W>::Permute(Word* dst, const Word* src, size_t num_primes,
                              size_t degree, unsigned int log_n,
                              uint32_t galois_factor) {
  for (size_t l = 0; l < num_primes; ++l) {
    const Word* src_limb = src + l * degree;
    Word* dst_limb = dst + l * degree;
    for (size_t x = 0; x < degree; ++x) {
      uint32_t x_rev =
          MontArith<W>::BitReverse(static_cast<Word>(x), log_n + 1) + 1;
      uint32_t src_rev = MontArith<W>::BitReverse(
          static_cast<Word>(x_rev * galois_factor - 1), log_n + 1);
      dst_limb[x] = src_limb[src_rev];
    }
  }
}

template <unsigned W>
void RefKeySwitch<W>::PermuteConjugate(Word* dst, const Word* src,
                                       size_t num_primes, size_t degree,
                                       unsigned int log_n) {
  uint32_t conj_gf = 2 * degree - 1;
  Permute(dst, src, num_primes, degree, log_n, conj_gf);
}

template <unsigned W>
RefTestKeys<W> RefKeySwitch<W>::MakeTestKeys(
    const Params<W>& params, const Tables<W>& tables, uint64_t seed,
    const std::vector<int>& rot_indices) {
  LocalSplitMix64 prng(seed);
  size_t degree = params.degree;
  unsigned int log_n = params.log_n;
  size_t num_total_primes = params.all_primes.size();
  size_t num_q = params.L;
  size_t num_aux = params.alpha;
  int beta = (num_q + num_aux - 1) / num_aux;

  RefTestKeys<W> keys;
  keys.degree = degree;
  keys.num_total_primes = num_total_primes;
  keys.beta = beta;

  // 1. Secret key s (sparse ternary with dense_hamming_weight)
  int hw = params.dense_hamming_weight;
  std::vector<int> pool(degree);
  std::iota(pool.begin(), pool.end(), 0);
  for (int i = 0; i < hw; ++i) {
    uint64_t r = prng.template NextUniform<uint64_t>(degree - i);
    std::swap(pool[i], pool[i + r]);
  }
  std::vector<int> ternary_vals(hw);
  for (int i = 0; i < hw; ++i) {
    ternary_vals[i] = prng.Next() & 1;
  }

  keys.secret_key.assign(num_total_primes * degree, 0);
  for (size_t l = 0; l < num_total_primes; ++l) {
    Word q = params.all_primes[l];
    for (int j = 0; j < hw; ++j) {
      keys.secret_key[l * degree + pool[j]] =
          (ternary_vals[j] == 0) ? 1 : (q - 1);
    }
  }

  // NTT on secret key to evaluation domain
  RefKernels<W>::NTT(keys.secret_key.data(), tables,
                     LimbRange{0, num_total_primes},
                     /*montgomery_conversion=*/true);

  // Compute s^2 in eval domain
  std::vector<Word> s_squared(num_total_primes * degree);
  for (size_t l = 0; l < num_total_primes; ++l) {
    Word q = params.all_primes[l];
    SignedWord q_inv = tables.q_inv[l];
    for (size_t c = 0; c < degree; ++c) {
      s_squared[l * degree + c] =
          MontArith<W>::MultMont(keys.secret_key[l * degree + c],
                                 keys.secret_key[l * degree + c], q, q_inv);
    }
  }

  auto prepare_evk = [&](RefEvaluationKey<W>& evk, const Word* enc_s,
                         const Word* target_s) {
    evk.beta = beta;
    evk.num_total_primes = num_total_primes;
    evk.degree = degree;
    evk.bx.resize(beta, std::vector<Word>(num_total_primes * degree));
    evk.ax.resize(beta, std::vector<Word>(num_total_primes * degree));

    for (int i = 0; i < beta; ++i) {
      // Sample ax uniform in [0, q)
      for (size_t l = 0; l < num_total_primes; ++l) {
        Word q = params.all_primes[l];
        for (size_t c = 0; c < degree; ++c) {
          evk.ax[i][l * degree + c] = prng.template NextUniform<Word>(q);
        }
      }

      // Sample discrete Gaussian error
      std::vector<int64_t> error(degree);
      double max_val = 6.0 * kErrorStdDev;
      for (size_t c = 0; c < degree; c += 2) {
        double u1 =
            (prng.Next() + 1.0) / (static_cast<double>(UINT64_MAX) + 2.0);
        double u2 =
            (prng.Next() + 1.0) / (static_cast<double>(UINT64_MAX) + 2.0);
        double r = std::sqrt(-2.0 * std::log(u1)) * kErrorStdDev;
        double theta = 2.0 * M_PI * u2;
        double z0 = r * std::cos(theta);
        double z1 = r * std::sin(theta);
        z0 = std::clamp(z0, -max_val, max_val);
        error[c] = std::llround(z0);
        if (c + 1 < degree) {
          z1 = std::clamp(z1, -max_val, max_val);
          error[c + 1] = std::llround(z1);
        }
      }

      std::vector<Word> ex(num_total_primes * degree);
      for (size_t l = 0; l < num_total_primes; ++l) {
        Word q = params.all_primes[l];
        for (size_t c = 0; c < degree; ++c) {
          int64_t val = error[c];
          ex[l * degree + c] = (val < 0) ? (q - static_cast<Word>(-val))
                                         : static_cast<Word>(val);
        }
      }
      RefKernels<W>::NTT(ex.data(), tables, LimbRange{0, num_total_primes},
                         /*montgomery_conversion=*/true);

      // bx = -ax * enc_s + ex
      for (size_t l = 0; l < num_total_primes; ++l) {
        Word q = params.all_primes[l];
        SignedWord q_inv = tables.q_inv[l];
        for (size_t c = 0; c < degree; ++c) {
          Word mult = MontArith<W>::MultMont(evk.ax[i][l * degree + c],
                                             enc_s[l * degree + c], q, q_inv);
          evk.bx[i][l * degree + c] =
              MontArith<W>::Sub(ex[l * degree + c], mult, q);
        }
      }

      // AddEvkPart: bx += target_s * p_prod for chunk
      size_t chunk_size =
          (i == beta - 1) ? (num_q - (beta - 1) * num_aux) : num_aux;
      for (size_t k = 0; k < chunk_size; ++k) {
        size_t l = i * num_aux + k;
        Word q = params.all_primes[l];
        SignedWord q_inv = tables.q_inv[l];
        Word p_prod_mont = ComputePProdMont<W>(params, q);
        for (size_t c = 0; c < degree; ++c) {
          Word term = MontArith<W>::MultMont(target_s[l * degree + c],
                                             p_prod_mont, q, q_inv);
          evk.bx[i][l * degree + c] =
              MontArith<W>::Add(evk.bx[i][l * degree + c], term, q);
        }
      }
    }
  };

  // 2. Relinearization key: target = s^2
  prepare_evk(keys.relin_key, keys.secret_key.data(), s_squared.data());

  // 3. Conjugation key: target = s permuted with -1
  std::vector<Word> s_conj(num_total_primes * degree);
  PermuteConjugate(s_conj.data(), keys.secret_key.data(), num_total_primes,
                   degree, log_n);
  prepare_evk(keys.conj_key, s_conj.data(), keys.secret_key.data());

  // 4. Rotation keys
  int half_degree = degree / 2;
  for (int rot_idx : rot_indices) {
    if (rot_idx <= 0 || rot_idx >= half_degree) continue;
    uint32_t gf = Tables<W>::PowMod(5, half_degree - rot_idx, 2 * degree);
    std::vector<Word> s_rot(num_total_primes * degree);
    Permute(s_rot.data(), keys.secret_key.data(), num_total_primes, degree,
            log_n, gf);
    prepare_evk(keys.rot_keys[rot_idx], s_rot.data(), keys.secret_key.data());
  }

  return keys;
}

template <unsigned W>
void RefKeySwitch<W>::KeySwitchNoModDown(
    Word* accum_bx, Word* accum_ax, const Word* in, const Word* in_bx,
    const Word* in_ax, const RefEvaluationKey<W>& evk, int level,
    const ModUpTableGroup<W>& mod_up_tables, const Tables<W>& general_tables,
    const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t num_aux = params.alpha;
  size_t degree = params.degree;
  size_t max_num_ter = params.ter_primes.size();
  size_t prime_offset = max_num_ter - num_ter;
  size_t padded_num_q = num_q + prime_offset;
  int beta = (padded_num_q + num_aux - 1) / num_aux;

  size_t accum_size = (num_q + num_aux) * degree;
  std::fill_n(accum_bx, accum_size, Word(0));
  std::fill_n(accum_ax, accum_size, Word(0));

  // 1. ModUp of `in` into beta buffers of shape [num_q + num_aux, degree]
  std::vector<std::vector<Word>> modup_results;
  RefModSwitch<W>::ModUp(modup_results, in, mod_up_tables, general_tables,
                         params);

  // 2. CPAccum over active digits
  for (int i = 0; i < beta; ++i) {
    size_t prime_index_end = std::min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) continue;

    for (size_t l = 0; l < num_q + num_aux; ++l) {
      size_t prime_idx =
          (l < num_q) ? (l + prime_offset)
                      : (params.all_primes.size() - num_aux + (l - num_q));
      Word q = params.all_primes[prime_idx];
      SignedWord q_inv = general_tables.q_inv[prime_idx];
      size_t evk_limb =
          (l < num_q) ? (l + prime_offset) : (params.L + (l - num_q));

      const Word* modup_limb = modup_results[i].data() + l * degree;
      const Word* evk_bx_limb = evk.bx[i].data() + evk_limb * degree;
      const Word* evk_ax_limb = evk.ax[i].data() + evk_limb * degree;
      Word* acc_bx_limb = accum_bx + l * degree;
      Word* acc_ax_limb = accum_ax + l * degree;

      for (size_t c = 0; c < degree; ++c) {
        Word modup_val = modup_limb[c];
        Word b_prod =
            MontArith<W>::MultMont(modup_val, evk_bx_limb[c], q, q_inv);
        acc_bx_limb[c] = MontArith<W>::Add(acc_bx_limb[c], b_prod, q);
        Word a_prod =
            MontArith<W>::MultMont(modup_val, evk_ax_limb[c], q, q_inv);
        acc_ax_limb[c] = MontArith<W>::Add(acc_ax_limb[c], a_prod, q);
      }
    }
  }

  // 3. Add p_prod * in_bx and p_prod * in_ax to first num_q limbs
  if (in_bx != nullptr) {
    for (size_t l = 0; l < num_q; ++l) {
      size_t prime_idx = l + prime_offset;
      Word q = params.all_primes[prime_idx];
      SignedWord q_inv = general_tables.q_inv[prime_idx];
      Word p_prod_mont = ComputePProdMont<W>(params, q);
      const Word* in_bx_limb = in_bx + l * degree;
      Word* acc_bx_limb = accum_bx + l * degree;
      for (size_t c = 0; c < degree; ++c) {
        Word prod =
            MontArith<W>::MultMont(in_bx_limb[c], p_prod_mont, q, q_inv);
        acc_bx_limb[c] = MontArith<W>::Add(acc_bx_limb[c], prod, q);
      }
    }
  }

  if (in_ax != nullptr) {
    for (size_t l = 0; l < num_q; ++l) {
      size_t prime_idx = l + prime_offset;
      Word q = params.all_primes[prime_idx];
      SignedWord q_inv = general_tables.q_inv[prime_idx];
      Word p_prod_mont = ComputePProdMont<W>(params, q);
      const Word* in_ax_limb = in_ax + l * degree;
      Word* acc_ax_limb = accum_ax + l * degree;
      for (size_t c = 0; c < degree; ++c) {
        Word prod =
            MontArith<W>::MultMont(in_ax_limb[c], p_prod_mont, q, q_inv);
        acc_ax_limb[c] = MontArith<W>::Add(acc_ax_limb[c], prod, q);
      }
    }
  }
}

template <unsigned W>
void RefKeySwitch<W>::KeySwitch(Word* out_bx, Word* out_ax, const Word* in_bx,
                                const Word* in_ax,
                                const RefEvaluationKey<W>& evk, int level,
                                const ModUpTableGroup<W>& mod_up_tables,
                                const ModDownTableGroup<W>& mod_down_tables,
                                const Tables<W>& general_tables,
                                const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t num_aux = params.alpha;
  size_t degree = params.degree;

  std::vector<Word> accum_bx((num_q + num_aux) * degree);
  std::vector<Word> accum_ax((num_q + num_aux) * degree);

  KeySwitchNoModDown(accum_bx.data(), accum_ax.data(), in_ax, in_bx,
                     /*in_ax=*/nullptr, evk, level, mod_up_tables,
                     general_tables, params);

  RefModSwitch<W>::ModDown(out_bx, accum_bx.data(), mod_down_tables,
                           general_tables, params);
  RefModSwitch<W>::ModDown(out_ax, accum_ax.data(), mod_down_tables,
                           general_tables, params);
}

template <unsigned W>
void RefKeySwitch<W>::Relinearize(Word* out_bx, Word* out_ax, const Word* in_bx,
                                  const Word* in_ax, const Word* in_rx,
                                  const RefEvaluationKey<W>& relin_key,
                                  int level,
                                  const ModUpTableGroup<W>& mod_up_tables,
                                  const ModDownTableGroup<W>& mod_down_tables,
                                  const Tables<W>& general_tables,
                                  const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t num_aux = params.alpha;
  size_t degree = params.degree;

  std::vector<Word> accum_bx((num_q + num_aux) * degree);
  std::vector<Word> accum_ax((num_q + num_aux) * degree);

  KeySwitchNoModDown(accum_bx.data(), accum_ax.data(), in_rx, in_bx, in_ax,
                     relin_key, level, mod_up_tables, general_tables, params);

  RefModSwitch<W>::ModDown(out_bx, accum_bx.data(), mod_down_tables,
                           general_tables, params);
  RefModSwitch<W>::ModDown(out_ax, accum_ax.data(), mod_down_tables,
                           general_tables, params);
}

template <unsigned W>
void RefKeySwitch<W>::RelinearizeRescale(
    Word* out_bx, Word* out_ax, const Word* in_bx, const Word* in_ax,
    const Word* in_rx, const RefEvaluationKey<W>& relin_key, int level,
    const ModUpTableGroup<W>& mod_up_tables,
    const ModDownRescaleTableGroup<W>& mod_down_rescale_tables,
    const Tables<W>& general_tables, const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t num_aux = params.alpha;
  size_t degree = params.degree;

  std::vector<Word> accum_bx((num_q + num_aux) * degree);
  std::vector<Word> accum_ax((num_q + num_aux) * degree);

  KeySwitchNoModDown(accum_bx.data(), accum_ax.data(), in_rx, in_bx, in_ax,
                     relin_key, level, mod_up_tables, general_tables, params);

  RefModSwitch<W>::ModDownAndRescale(
      out_bx, accum_bx.data(), mod_down_rescale_tables, general_tables, params);
  RefModSwitch<W>::ModDownAndRescale(
      out_ax, accum_ax.data(), mod_down_rescale_tables, general_tables, params);
}

template <unsigned W>
void RefKeySwitch<W>::HRot(Word* out_bx, Word* out_ax, const Word* in_bx,
                           const Word* in_ax,
                           const RefEvaluationKey<W>& rot_key, int level,
                           int rot_dist,
                           const ModUpTableGroup<W>& mod_up_tables,
                           const ModDownTableGroup<W>& mod_down_tables,
                           const Tables<W>& general_tables,
                           const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t degree = params.degree;
  unsigned int log_n = params.log_n;

  std::vector<Word> tmp_bx(num_q * degree);
  std::vector<Word> tmp_ax(num_q * degree);

  KeySwitch(tmp_bx.data(), tmp_ax.data(), in_bx, in_ax, rot_key, level,
            mod_up_tables, mod_down_tables, general_tables, params);

  uint32_t gf = Tables<W>::PowMod(5, rot_dist, 2 * degree);
  Permute(out_bx, tmp_bx.data(), num_q, degree, log_n, gf);
  Permute(out_ax, tmp_ax.data(), num_q, degree, log_n, gf);
}

template <unsigned W>
void RefKeySwitch<W>::HConj(Word* out_bx, Word* out_ax, const Word* in_bx,
                            const Word* in_ax,
                            const RefEvaluationKey<W>& conj_key, int level,
                            const ModUpTableGroup<W>& mod_up_tables,
                            const ModDownTableGroup<W>& mod_down_tables,
                            const Tables<W>& general_tables,
                            const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t degree = params.degree;
  unsigned int log_n = params.log_n;

  std::vector<Word> tmp_bx(num_q * degree);
  std::vector<Word> tmp_ax(num_q * degree);

  KeySwitch(tmp_bx.data(), tmp_ax.data(), in_bx, in_ax, conj_key, level,
            mod_up_tables, mod_down_tables, general_tables, params);

  PermuteConjugate(out_bx, tmp_bx.data(), num_q, degree, log_n);
  PermuteConjugate(out_ax, tmp_ax.data(), num_q, degree, log_n);
}

template <unsigned W>
void RefKeySwitch<W>::HRotAdd(
    Word* out_bx, Word* out_ax, const Word* in1_bx, const Word* in1_ax,
    const Word* in2_bx, const Word* in2_ax, const RefEvaluationKey<W>& rot_key,
    int level, int rot_dist, const ModUpTableGroup<W>& mod_up_tables,
    const ModDownTableGroup<W>& mod_down_tables,
    const Tables<W>& general_tables, const Params<W>& params) {
  size_t num_main = params.level_config[level].first;
  size_t num_ter = params.level_config[level].second;
  size_t num_q = num_main + num_ter;
  size_t degree = params.degree;
  size_t max_num_ter = params.ter_primes.size();
  size_t prime_offset = max_num_ter - num_ter;

  HRot(out_bx, out_ax, in1_bx, in1_ax, rot_key, level, rot_dist, mod_up_tables,
       mod_down_tables, general_tables, params);

  for (size_t l = 0; l < num_q; ++l) {
    size_t prime_idx = l + prime_offset;
    Word q = params.all_primes[prime_idx];
    for (size_t c = 0; c < degree; ++c) {
      out_bx[l * degree + c] =
          MontArith<W>::Add(out_bx[l * degree + c], in2_bx[l * degree + c], q);
      out_ax[l * degree + c] =
          MontArith<W>::Add(out_ax[l * degree + c], in2_ax[l * degree + c], q);
    }
  }
}

template <unsigned W>
void RefKeySwitch<W>::Encrypt(Word* bx, Word* ax, const Word* mx,
                              const Word* sx, size_t num_q, size_t prime_offset,
                              size_t degree, const Params<W>& params,
                              const Tables<W>& general_tables, uint64_t seed) {
  LocalSplitMix64 prng(seed);
  for (size_t l = 0; l < num_q; ++l) {
    size_t prime_idx = l + prime_offset;
    Word q = params.all_primes[prime_idx];
    for (size_t c = 0; c < degree; ++c) {
      ax[l * degree + c] = prng.template NextUniform<Word>(q);
    }
  }

  // Discrete Gaussian error
  std::vector<int64_t> error(degree);
  double max_val = 6.0 * kErrorStdDev;
  for (size_t c = 0; c < degree; c += 2) {
    double u1 = (prng.Next() + 1.0) / (static_cast<double>(UINT64_MAX) + 2.0);
    double u2 = (prng.Next() + 1.0) / (static_cast<double>(UINT64_MAX) + 2.0);
    double r = std::sqrt(-2.0 * std::log(u1)) * kErrorStdDev;
    double theta = 2.0 * M_PI * u2;
    double z0 = r * std::cos(theta);
    double z1 = r * std::sin(theta);
    z0 = std::clamp(z0, -max_val, max_val);
    error[c] = std::llround(z0);
    if (c + 1 < degree) {
      z1 = std::clamp(z1, -max_val, max_val);
      error[c + 1] = std::llround(z1);
    }
  }

  std::vector<Word> ex(num_q * degree);
  for (size_t l = 0; l < num_q; ++l) {
    size_t prime_idx = l + prime_offset;
    Word q = params.all_primes[prime_idx];
    for (size_t c = 0; c < degree; ++c) {
      int64_t val = error[c];
      ex[l * degree + c] =
          (val < 0) ? (q - static_cast<Word>(-val)) : static_cast<Word>(val);
    }
  }

  RefKernels<W>::NTT(ex.data(), general_tables,
                     LimbRange{prime_offset, prime_offset + num_q},
                     /*montgomery_conversion=*/true);

  std::vector<Word> mx_eval(num_q * degree);
  std::copy(mx, mx + num_q * degree, mx_eval.begin());
  RefKernels<W>::NTT(mx_eval.data(), general_tables,
                     LimbRange{prime_offset, prime_offset + num_q},
                     /*montgomery_conversion=*/true);

  // bx = -ax * sx + mx + ex
  for (size_t l = 0; l < num_q; ++l) {
    size_t prime_idx = l + prime_offset;
    Word q = params.all_primes[prime_idx];
    SignedWord q_inv = general_tables.q_inv[prime_idx];
    const Word* sx_limb = sx + prime_idx * degree;
    for (size_t c = 0; c < degree; ++c) {
      Word mult =
          MontArith<W>::MultMont(ax[l * degree + c], sx_limb[c], q, q_inv);
      Word res = MontArith<W>::Sub(mx_eval[l * degree + c], mult, q);
      bx[l * degree + c] = MontArith<W>::Add(res, ex[l * degree + c], q);
    }
  }
}

template <unsigned W>
void RefKeySwitch<W>::Decrypt(Word* mx, const Word* bx, const Word* ax,
                              const Word* sx, size_t num_q, size_t prime_offset,
                              size_t degree, const Params<W>& params,
                              const Tables<W>& general_tables) {
  for (size_t l = 0; l < num_q; ++l) {
    size_t prime_idx = l + prime_offset;
    Word q = params.all_primes[prime_idx];
    SignedWord q_inv = general_tables.q_inv[prime_idx];
    const Word* sx_limb = sx + prime_idx * degree;
    for (size_t c = 0; c < degree; ++c) {
      Word mult =
          MontArith<W>::MultMont(ax[l * degree + c], sx_limb[c], q, q_inv);
      mx[l * degree + c] = MontArith<W>::Add(bx[l * degree + c], mult, q);
    }
  }

  RefKernels<W>::INTT(mx, general_tables,
                      LimbRange{prime_offset, prime_offset + num_q},
                      /*inv_default_true=*/true);
}

template class RefKeySwitch<32>;
template class RefKeySwitch<64>;

}  // namespace mlir::heir::cheddar_ref
