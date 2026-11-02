#include "lib/Utils/CheddarRef/RefModSwitch.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

namespace {

template <unsigned W>
std::vector<typename MontArithTraits<W>::Word> GetLevelPrimes(
    const Params<W>& params, int level, int num_aux = 0) {
  using Word = typename MontArithTraits<W>::Word;
  int num_main = 0;
  int num_ter = 0;
  if (level == -1) {
    num_main = params.level_config.at(0).first - params.additional_base.first;
    num_ter = params.level_config.at(0).second - params.additional_base.second;
  } else {
    num_main = params.level_config.at(level).first;
    num_ter = params.level_config.at(level).second;
  }

  std::vector<Word> primes;
  for (int i = 0; i < num_ter; ++i) {
    primes.push_back(params.ter_primes[num_ter - 1 - i]);
  }
  for (int i = 0; i < num_main; ++i) {
    primes.push_back(params.main_primes[i]);
  }
  for (int i = 0; i < num_aux; ++i) {
    primes.push_back(params.aux_primes[i]);
  }
  return primes;
}

template <unsigned W>
Tables<W> MakeLevelTables(
    const Tables<W>& general_tables,
    const std::vector<typename MontArithTraits<W>::Word>& level_primes) {
  using Word = typename MontArithTraits<W>::Word;
  Tables<W> t;
  t.degree = general_tables.degree;
  t.primes = level_primes;
  size_t num_primes = level_primes.size();
  t.twiddle_limbs = num_primes;
  t.q_inv.resize(num_primes);
  t.R_mod_q.resize(num_primes);
  t.R2_mod_q.resize(num_primes);
  t.N_inv.resize(num_primes);
  t.N_inv_mont.resize(num_primes);
  t.psi_rev_mont.resize(num_primes * t.degree);
  t.psi_inv_rev_mont.resize(num_primes * t.degree);

  for (size_t l = 0; l < num_primes; ++l) {
    Word p = level_primes[l];
    auto it = std::find(general_tables.primes.begin(),
                        general_tables.primes.end(), p);
    assert(it != general_tables.primes.end() &&
           "Prime not found in general_tables");
    size_t orig_idx = std::distance(general_tables.primes.begin(), it);

    t.q_inv[l] = general_tables.q_inv[orig_idx];
    t.R_mod_q[l] = general_tables.R_mod_q[orig_idx];
    t.R2_mod_q[l] = general_tables.R2_mod_q[orig_idx];
    t.N_inv[l] = general_tables.N_inv[orig_idx];
    t.N_inv_mont[l] = general_tables.N_inv_mont[orig_idx];

    if (orig_idx < general_tables.twiddle_limbs) {
      std::copy_n(general_tables.psi_rev_mont.data() + orig_idx * t.degree,
                  t.degree, t.psi_rev_mont.data() + l * t.degree);
      std::copy_n(general_tables.psi_inv_rev_mont.data() + orig_idx * t.degree,
                  t.degree, t.psi_inv_rev_mont.data() + l * t.degree);
    } else {
      Word deg = static_cast<Word>(t.degree);
      Word psi = Tables<W>::FindPrimitiveMthRoot(2 * deg, p);
      Word psi_inv = Tables<W>::InvMod(psi, p);

      std::vector<Word> psi_rev(t.degree);
      std::vector<Word> psi_inv_rev(t.degree);
      psi_rev[0] = 1;
      psi_inv_rev[0] = 1;
      for (size_t j = 1; j < t.degree; ++j) {
        psi_rev[j] = Tables<W>::MultMod(psi_rev[j - 1], psi, p);
        psi_inv_rev[j] = Tables<W>::MultMod(psi_inv_rev[j - 1], psi_inv, p);
      }

      int log_slots = 0;
      while ((1ULL << log_slots) < t.degree) ++log_slots;
      for (size_t i = 0; i < t.degree; ++i) {
        size_t j = MontArith<W>::BitReverse(i, log_slots);
        if (i < j) {
          std::swap(psi_rev[i], psi_rev[j]);
          std::swap(psi_inv_rev[i], psi_inv_rev[j]);
        }
      }

      for (size_t j = 0; j < t.degree; ++j) {
        t.psi_rev_mont[l * t.degree + j] = MontArith<W>::ToMont(psi_rev[j], p);
        t.psi_inv_rev_mont[l * t.degree + j] =
            MontArith<W>::ToMont(psi_inv_rev[j], p);
      }
    }
  }
  return t;
}

}  // namespace

template <unsigned W>
void RefModSwitch<W>::ModDownEpilogue(
    Word* dst, const Word* ntt_bconv, const Word* inv_prime_prod,
    size_t dst_len, size_t degree, const Word* dst_primes,
    const SignedWord* dst_q_inv, const Word* src2, const Word* src2_padding,
    size_t pad_start, size_t pad_end) {
  // dst[l, i] = ((src2_exists ? (src2[l - pad_start, i] * src2_padding[l -
  // pad_start]) : 0) - ntt_bconv[l, i]) * inv_prime_prod[l] mod q_l. Note: in
  // Cheddar's NTTPhase2ForModDown, src2 has length pad_len = pad_end -
  // pad_start. For l outside [pad_start, pad_end), src2 does not exist for limb
  // l, so term is 0 - ntt_bconv[l, i]. For l inside [pad_start, pad_end), src2
  // is indexed at (l - pad_start). If src2_padding is provided, src2[l -
  // pad_start, i] is multiplied by src2_padding[l - pad_start] in Montgomery
  // form.
  for (size_t l = 0; l < dst_len; ++l) {
    Word q = dst_primes[l];
    SignedWord q_inv = dst_q_inv[l];
    Word inv_p = inv_prime_prod[l];

    bool has_src2 = (src2 != nullptr && l >= pad_start && l < pad_end);
    size_t src2_limb_idx = has_src2 ? (l - pad_start) : 0;
    Word pad_const =
        (has_src2 && src2_padding != nullptr) ? src2_padding[src2_limb_idx] : 1;

    for (size_t i = 0; i < degree; ++i) {
      Word term1 = 0;
      if (has_src2) {
        Word s2_val = src2[src2_limb_idx * degree + i];
        if (src2_padding != nullptr) {
          term1 = MontArith<W>::MultMont(s2_val, pad_const, q, q_inv);
        } else {
          term1 = s2_val;
        }
      }
      Word bconv_val = ntt_bconv[l * degree + i];
      Word diff = MontArith<W>::Sub(term1, bconv_val, q);
      Word res = MontArith<W>::MultMont(diff, inv_p, q, q_inv);
      dst[l * degree + i] = res;
    }
  }
}

template <unsigned W>
void RefModSwitch<W>::PseudoModUp(Word* dst, const Word* src,
                                  const PseudoModUpTableGroup<W>& tables,
                                  const Tables<W>& general_tables,
                                  const Params<W>& params, int level) {
  std::vector<Word> level_primes = GetLevelPrimes(params, level);
  size_t num_q_primes = level_primes.size();
  size_t degree = params.degree;
  for (size_t l = 0; l < num_q_primes; ++l) {
    Word q = level_primes[l];
    SignedWord q_inv = MontArith<W>::ComputeQInv(q);
    Word p_prod_val = tables.p_prod[l];
    for (size_t i = 0; i < degree; ++i) {
      dst[l * degree + i] =
          MontArith<W>::MultMont(src[l * degree + i], p_prod_val, q, q_inv);
    }
  }
}

template <unsigned W>
void RefModSwitch<W>::ModUp(std::vector<std::vector<Word>>& dsts,
                            const Word* src, const ModUpTableGroup<W>& tables,
                            const Tables<W>& general_tables,
                            const Params<W>& params) {
  size_t degree = params.degree;
  int num_q_primes = tables.num_q_primes;
  int num_aux = tables.num_aux;
  int beta = tables.beta;

  dsts.resize(beta);
  for (int i = 0; i < beta; ++i) {
    dsts[i].assign((num_q_primes + num_aux) * degree, 0);
  }

  // 1. INTT with mult_const_normalize
  // Create per-level tables for transforming `src`
  std::vector<Word> level_primes =
      GetLevelPrimes(params, tables.level, num_aux);
  Tables<W> level_tables = MakeLevelTables(general_tables, level_primes);

  std::vector<Word> src_intt(num_q_primes * degree);
  std::copy(src, src + num_q_primes * degree, src_intt.begin());

  RefKernels<W>::INTTAndMultConst(
      src_intt.data(), level_tables, tables.mod_up1.data(),
      LimbRange{0, static_cast<size_t>(num_q_primes)}, /*normalize=*/true);

  // Cast src_intt to SignedWord* for BConv
  const SignedWord* src_intt_signed =
      reinterpret_cast<const SignedWord*>(src_intt.data());

  // 2. Loop over beta digits
  for (int i = 0; i < beta; ++i) {
    const auto& digit = tables.digits[i];
    int prime_index_start = digit.prime_index_start;
    int prime_index_end = digit.prime_index_end;
    int src_len = digit.src_len;
    int dst_len = digit.dst_len;

    std::vector<Word>& dst_i = dsts[i];

    // Copy verbatim limbs from src to dst_i
    if (src_len > 0) {
      std::memcpy(dst_i.data() + prime_index_start * degree,
                  src + prime_index_start * degree,
                  src_len * degree * sizeof(Word));
    }

    // BConv
    // Destination primes for BConv are level_primes excluding
    // [prime_index_start, prime_index_end) plus aux primes (which are already
    // in level_primes at the end). In Cheddar's kernel::ModSwitchMatrixMult:
    // dst_i has shape [num_q_primes + num_aux, degree].
    // dst_y_position skips [prime_index_start, prime_index_end).
    std::vector<Word> bconv_dst_primes;
    std::vector<SignedWord> bconv_dst_inv;
    for (int l = 0; l < num_q_primes + num_aux; ++l) {
      if (l >= prime_index_start && l < prime_index_end) continue;
      bconv_dst_primes.push_back(level_primes[l]);
      bconv_dst_inv.push_back(level_tables.q_inv[l]);
    }

    // Temporary destination buffer for BConv of shape [dst_len, degree]
    std::vector<Word> bconv_out(dst_len * degree);
    RefBConv<W>::BConv(
        bconv_out.data(), src_intt_signed + prime_index_start * degree,
        digit.table.data(), src_len, dst_len, degree, bconv_dst_primes.data(),
        bconv_dst_inv.data(), /*skip_start=*/0, /*skip_end=*/0,
        /*periodic_recenter=*/true);

    // Place BConv output back into dst_i at limbs outside [prime_index_start,
    // prime_index_end)
    size_t bconv_m = 0;
    for (int l = 0; l < num_q_primes + num_aux; ++l) {
      if (l >= prime_index_start && l < prime_index_end) continue;
      std::memcpy(dst_i.data() + l * degree,
                  bconv_out.data() + bconv_m * degree, degree * sizeof(Word));
      bconv_m++;
    }

    // 3. NTT on dst_i for limbs outside [prime_index_start, prime_index_end)
    // with montgomery_conversion == false.
    RefKernels<W>::NTTForModUp(
        dst_i.data(), level_tables, prime_index_start, prime_index_end,
        LimbRange{0, static_cast<size_t>(num_q_primes + num_aux)});
  }
}

template <unsigned W>
void RefModSwitch<W>::ModDown(Word* dst, const Word* src,
                              const ModDownTableGroup<W>& tables,
                              const Tables<W>& general_tables,
                              const Params<W>& params) {
  size_t degree = params.degree;
  int num_q_primes = tables.num_q_primes;
  int num_aux = tables.num_aux;
  int src_len = tables.src_len;  // num_aux
  int dst_len = tables.dst_len;  // num_q_primes

  std::vector<Word> level_primes =
      GetLevelPrimes(params, tables.level, num_aux);
  Tables<W> level_tables = MakeLevelTables(general_tables, level_primes);

  // 1. INTT on aux primes (limbs in [num_q_primes, num_q_primes + num_aux))
  // with mod_down1 normalization
  std::vector<Word> aux_intt(src_len * degree);
  std::copy(src + num_q_primes * degree,
            src + (num_q_primes + num_aux) * degree, aux_intt.begin());

  RefKernels<W>::INTTAndMultConst(
      aux_intt.data(), level_tables, tables.mod_down1.data(),
      LimbRange{static_cast<size_t>(num_q_primes),
                static_cast<size_t>(num_q_primes + num_aux)},
      /*normalize=*/true);

  const SignedWord* aux_intt_signed =
      reinterpret_cast<const SignedWord*>(aux_intt.data());

  // 2. BConv from aux primes to Q primes
  std::vector<Word> bconv_out(dst_len * degree);
  RefBConv<W>::BConv(
      bconv_out.data(), aux_intt_signed, tables.mod_down2.data(), src_len,
      dst_len, degree, level_primes.data(), level_tables.q_inv.data(),
      /*skip_start=*/0, /*skip_end=*/0, /*periodic_recenter=*/true);

  // 3. NTT on bconv_out with montgomery_conversion == false
  RefKernels<W>::NTT(bconv_out.data(), level_tables,
                     LimbRange{0, static_cast<size_t>(dst_len)},
                     /*montgomery_conversion=*/false);

  // 4. ModDown epilogue: dst = (src - bconv) * inv_prime_prod mod q
  // For ModDown: src2 is src[0 ... num_q_primes * degree], pad_start=0,
  // pad_end=dst_len. src2_padding is nullptr.
  ModDownEpilogue(dst, bconv_out.data(), tables.inv_prime_prod.data(), dst_len,
                  degree, level_primes.data(), level_tables.q_inv.data(), src,
                  nullptr, /*pad_start=*/0, /*pad_end=*/dst_len);
}

template <unsigned W>
void RefModSwitch<W>::Rescale(Word* dst, const Word* src,
                              const RescaleTableGroup<W>& tables,
                              const Tables<W>& general_tables,
                              const Params<W>& params) {
  size_t degree = params.degree;
  int src_len = tables.src_len;
  int dst_len = tables.dst_len;

  std::vector<Word> curr_level_primes =
      GetLevelPrimes(params, tables.level, /*num_aux=*/0);
  std::vector<Word> next_level_primes =
      GetLevelPrimes(params, tables.level - 1, /*num_aux=*/0);

  Tables<W> curr_tables = MakeLevelTables(general_tables, curr_level_primes);
  Tables<W> next_tables = MakeLevelTables(general_tables, next_level_primes);

  // Determine which limbs of src are being INTT'd
  // In Cheddar:
  // int num_src_aux = 0;
  // np_src = LevelToNP(level_, 0);
  // np_dst = LevelToNP(level_ - 1, 0);
  // np_non_intt = (min(main), min(ter), 0);
  // INTT is performed on limbs [np_non_intt.GetNumTotal(),
  // np_src.GetNumTotal())
  int num_main = params.level_config.at(tables.level).first;
  int num_ter = params.level_config.at(tables.level).second;
  int next_main = params.level_config.at(tables.level - 1).first;
  int next_ter = params.level_config.at(tables.level - 1).second;

  int ter_diff = num_ter - next_ter;
  int main_diff = num_main - next_main;

  size_t intt_src_offset = 0;
  if (ter_diff > 0) {
    // ter decreases: INTT on first ter_diff limbs of src
    intt_src_offset = 0;
  } else if (main_diff > 0) {
    // main decreases: INTT on last main_diff limbs of src
    intt_src_offset = (num_ter + num_main - main_diff) * degree;
  }

  std::vector<Word> src_intt(src_len * degree);
  std::copy(src + intt_src_offset, src + intt_src_offset + src_len * degree,
            src_intt.begin());

  size_t intt_start = (ter_diff > 0) ? 0 : (num_ter + num_main - main_diff);
  size_t intt_end = intt_start + src_len;
  RefKernels<W>::INTTAndMultConst(src_intt.data(), curr_tables,
                                  tables.rescale1.data(),
                                  LimbRange{intt_start, intt_end},
                                  /*normalize=*/true);

  const SignedWord* src_intt_signed =
      reinterpret_cast<const SignedWord*>(src_intt.data());

  // 2. BConv to next_level_primes
  std::vector<Word> bconv_out(dst_len * degree);
  RefBConv<W>::BConv(
      bconv_out.data(), src_intt_signed, tables.rescale2.data(), src_len,
      dst_len, degree, next_level_primes.data(), next_tables.q_inv.data(),
      /*skip_start=*/0, /*skip_end=*/0, /*periodic_recenter=*/true);

  // 3. NTT on bconv_out
  RefKernels<W>::NTT(bconv_out.data(), next_tables,
                     LimbRange{0, static_cast<size_t>(dst_len)},
                     /*montgomery_conversion=*/false);

  // 4. ModDown epilogue:
  // int src2_offset = Max(0, np_src.num_ter_ - np_dst.num_ter_);
  // DvConstView<word> src2(src_view.data() + src2_offset * degree, pad_len *
  // degree, 0);
  int src2_offset = std::max(0, num_ter - next_ter);
  const Word* src2 = src + src2_offset * degree;

  ModDownEpilogue(dst, bconv_out.data(), tables.inv_prime_prod.data(), dst_len,
                  degree, next_level_primes.data(), next_tables.q_inv.data(),
                  src2, tables.src2_padding.data(), tables.pad_start,
                  tables.pad_end);
}

template <unsigned W>
void RefModSwitch<W>::ModDownAndRescale(
    Word* dst, const Word* src, const ModDownRescaleTableGroup<W>& tables,
    const Tables<W>& general_tables, const Params<W>& params) {
  size_t degree = params.degree;
  int src_len = tables.src_len;
  int dst_len = tables.dst_len;
  int num_aux = params.alpha;

  std::vector<Word> curr_level_primes =
      GetLevelPrimes(params, tables.level, num_aux);
  std::vector<Word> next_level_primes =
      GetLevelPrimes(params, tables.level - 1, /*num_aux=*/0);

  Tables<W> next_tables = MakeLevelTables(general_tables, next_level_primes);

  int num_main = params.level_config.at(tables.level).first;
  int num_ter = params.level_config.at(tables.level).second;
  int next_main = params.level_config.at(tables.level - 1).first;
  int next_ter = params.level_config.at(tables.level - 1).second;

  int ter_diff = num_ter - next_ter;
  int main_diff = num_main - next_main;

  // In ModDownAndRescale:
  // INTT is performed on mod_down_rescale_src_primes:
  // if ter_diff > 0: rescale_src_primes (first ter_diff) + P primes (last
  // num_aux) if main_diff > 0: rescale_src_primes (last main_diff of Q) + P
  // primes (last num_aux)
  std::vector<Word> src_intt(src_len * degree);
  std::vector<Word> intt_primes;

  if (ter_diff > 0) {
    std::copy(src, src + ter_diff * degree, src_intt.begin());
    std::copy(src + (num_ter + num_main) * degree,
              src + (num_ter + num_main + num_aux) * degree,
              src_intt.begin() + ter_diff * degree);
    intt_primes.assign(curr_level_primes.begin(),
                       curr_level_primes.begin() + ter_diff);
    intt_primes.insert(intt_primes.end(),
                       curr_level_primes.begin() + num_ter + num_main,
                       curr_level_primes.end());
  } else {
    size_t q_offset = (num_ter + num_main - main_diff) * degree;
    std::copy(src + q_offset, src + q_offset + main_diff * degree,
              src_intt.begin());
    std::copy(src + (num_ter + num_main) * degree,
              src + (num_ter + num_main + num_aux) * degree,
              src_intt.begin() + main_diff * degree);
    intt_primes.assign(
        curr_level_primes.begin() + num_ter + num_main - main_diff,
        curr_level_primes.begin() + num_ter + num_main);
    intt_primes.insert(intt_primes.end(),
                       curr_level_primes.begin() + num_ter + num_main,
                       curr_level_primes.end());
  }

  Tables<W> curr_tables = MakeLevelTables(general_tables, curr_level_primes);

  size_t q_start = (ter_diff > 0) ? 0 : (num_ter + num_main - main_diff);
  size_t q_count = (ter_diff > 0) ? ter_diff : main_diff;
  size_t q_end = q_start + q_count;

  RefKernels<W>::INTTAndMultConst(src_intt.data(), curr_tables,
                                  tables.mod_down_rescale1.data(),
                                  LimbRange{q_start, q_end},
                                  /*normalize=*/true);

  size_t p_start = num_ter + num_main;
  size_t p_end = p_start + num_aux;
  RefKernels<W>::INTTAndMultConst(
      src_intt.data() + q_count * degree, curr_tables,
      tables.mod_down_rescale1.data() + q_count, LimbRange{p_start, p_end},
      /*normalize=*/true);

  const SignedWord* src_intt_signed =
      reinterpret_cast<const SignedWord*>(src_intt.data());

  // 2. BConv to next_level_primes
  std::vector<Word> bconv_out(dst_len * degree);
  RefBConv<W>::BConv(bconv_out.data(), src_intt_signed,
                     tables.mod_down_rescale2.data(), src_len, dst_len, degree,
                     next_level_primes.data(), next_tables.q_inv.data(),
                     /*skip_start=*/0, /*skip_end=*/0,
                     /*periodic_recenter=*/true);

  // 3. NTT on bconv_out
  RefKernels<W>::NTT(bconv_out.data(), next_tables,
                     LimbRange{0, static_cast<size_t>(dst_len)},
                     /*montgomery_conversion=*/false);

  // 4. ModDown epilogue:
  int src2_offset = std::max(0, num_ter - next_ter);
  const Word* src2 = src + src2_offset * degree;

  ModDownEpilogue(dst, bconv_out.data(), tables.inv_prime_prod.data(), dst_len,
                  degree, next_level_primes.data(), next_tables.q_inv.data(),
                  src2, tables.src2_padding.data(), tables.pad_start,
                  tables.pad_end);
}

template class RefModSwitch<32>;
template class RefModSwitch<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
