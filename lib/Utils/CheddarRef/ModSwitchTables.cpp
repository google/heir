#include "lib/Utils/CheddarRef/ModSwitchTables.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

namespace {

// Helper to extract prime vector for a level in Cheddar's ordering
template <unsigned W>
std::vector<typename MontArithTraits<W>::Word> GetLevelPrimes(
    const Params<W>& params, int level, int num_aux = 0) {
  using Word = typename MontArithTraits<W>::Word;
  // level_config has (num_main, num_ter)
  int num_main = 0;
  int num_ter = 0;
  if (level == -1) {
    num_main = params.level_config.at(0).first - params.additional_base.first;
    num_ter = params.level_config.at(0).second - params.additional_base.second;
  } else {
    num_main = params.level_config.at(level).first;
    num_ter = params.level_config.at(level).second;
  }

  assert(num_ter <= static_cast<int>(params.ter_primes.size()) &&
         "num_ter exceeds ter_primes.size()");
  assert(num_main <= static_cast<int>(params.main_primes.size()) &&
         "num_main exceeds main_primes.size()");
  assert(num_aux <= static_cast<int>(params.aux_primes.size()) &&
         "num_aux exceeds aux_primes.size()");
  std::vector<Word> primes;
  // ter_primes[num_ter - 1 ... 0] (matching Cheddar's GetPrimeVector)
  for (int i = 0; i < num_ter; ++i) {
    primes.push_back(params.ter_primes[num_ter - 1 - i]);
  }
  // main_primes[0 ... num_main - 1]
  for (int i = 0; i < num_main; ++i) {
    primes.push_back(params.main_primes[i]);
  }
  // aux_primes[0 ... num_aux - 1]
  for (int i = 0; i < num_aux; ++i) {
    primes.push_back(params.aux_primes[i]);
  }
  return primes;
}

}  // namespace

template <unsigned W>
void ModSwitchTables<W>::PopulateModSwitchConstants(
    std::vector<Word>& const1, std::vector<SignedWord>& const2,
    const std::vector<Word>& src_primes, const std::vector<Word>& dst_primes,
    size_t degree, int restore_start, int restore_end) {
  int src_len = src_primes.size();
  int dst_len = dst_primes.size();

  const1.assign(src_len, 1);
  const2.assign(src_len * dst_len, 1);

  for (int i = 0; i < src_len; i++) {
    Word modulus = src_primes[i];
    for (int j = 0; j < src_len; j++) {
      if (j == i) continue;
      const1[i] = Tables<W>::MultMod(const1[i], src_primes[j], modulus);
    }
    const1[i] = Tables<W>::MultMod(const1[i], static_cast<Word>(degree),
                                   modulus);  // for inverse N of INTT
    const1[i] = Tables<W>::InvMod(const1[i], modulus);

    // BitPacker extension for fused epilogue
    for (int j = restore_start; j < restore_end; j++) {
      const1[i] = Tables<W>::MultMod(const1[i], dst_primes[j], modulus);
    }
  }

  for (int m = 0; m < dst_len; m++) {
    Word mod_prime = dst_primes[m];
    for (int i = 0; i < src_len; i++) {
      Word accum = 1;
      for (int j = 0; j < src_len; j++) {
        if (i == j) continue;
        accum = Tables<W>::MultMod(accum, src_primes[j], mod_prime);
      }
      accum = MontArith<W>::ToMont(accum, mod_prime);
      accum = MontArith<W>::ToMont(accum, mod_prime);
      const2[m * src_len + i] = MontArith<W>::Normalize(accum, mod_prime);
    }
  }
}

template <unsigned W>
void ModSwitchTables<W>::PopulateModDownEpilogueConstants(
    std::vector<Word>& inv_prime_prod, std::vector<Word>& padding,
    const std::vector<Word>& src_primes, const std::vector<Word>& dst_primes,
    int restore_start, int restore_end) {
  int src_len = src_primes.size();
  int dst_len = dst_primes.size();
  int restore_len = restore_end - restore_start;

  inv_prime_prod.assign(dst_len, 1);
  for (int i = 0; i < dst_len; i++) {
    Word mod_prime = dst_primes[i];
    for (int j = 0; j < src_len; j++) {
      inv_prime_prod[i] =
          Tables<W>::MultMod(inv_prime_prod[i], src_primes[j], mod_prime);
    }
    inv_prime_prod[i] = Tables<W>::InvMod(inv_prime_prod[i], mod_prime);
    inv_prime_prod[i] = MontArith<W>::ToMont(inv_prime_prod[i], mod_prime);
  }

  padding.assign(dst_len - restore_len, 1);
  for (int i = 0; i < dst_len - restore_len; i++) {
    int mod_prime_index = i;
    if (i >= restore_start) {
      mod_prime_index += restore_len;
    }
    Word mod_prime = dst_primes[mod_prime_index];
    for (int j = restore_start; j < restore_end; j++) {
      padding[i] = Tables<W>::MultMod(padding[i], dst_primes[j], mod_prime);
    }
    padding[i] = MontArith<W>::ToMont(padding[i], mod_prime);
  }
}

template <unsigned W>
PseudoModUpTableGroup<W> ModSwitchTables<W>::GeneratePseudoModUp(
    const Params<W>& params, int level) {
  PseudoModUpTableGroup<W> grp;
  int num_main =
      (level == -1)
          ? (params.level_config.at(0).first - params.additional_base.first)
          : params.level_config.at(level).first;
  int num_ter =
      (level == -1)
          ? (params.level_config.at(0).second - params.additional_base.second)
          : params.level_config.at(level).second;
  int num_q = num_main + num_ter;
  int num_aux = (level == -1) ? (params.level_config.at(0).first +
                                 params.level_config.at(0).second -
                                 params.additional_base.first -
                                 params.additional_base.second)
                              : params.alpha;

  std::vector<Word> level_primes = GetLevelPrimes(params, level, num_aux);
  grp.p_prod.assign(num_q, 1);
  for (int i = 0; i < num_q; ++i) {
    Word mod_prime = level_primes[i];
    for (int j = 0; j < num_aux; ++j) {
      grp.p_prod[i] =
          Tables<W>::MultMod(grp.p_prod[i], level_primes[num_q + j], mod_prime);
    }
    grp.p_prod[i] = MontArith<W>::ToMont(grp.p_prod[i], mod_prime);
  }
  return grp;
}

template <unsigned W>
ModUpTableGroup<W> ModSwitchTables<W>::GenerateModUp(const Params<W>& params,
                                                     int level) {
  ModUpTableGroup<W> grp;
  grp.level = level;
  int num_aux = (level == -1) ? (params.level_config.at(0).first +
                                 params.level_config.at(0).second -
                                 params.additional_base.first -
                                 params.additional_base.second)
                              : params.alpha;
  grp.num_aux = num_aux;

  int num_main =
      (level == -1)
          ? (params.level_config.at(0).first - params.additional_base.first)
          : params.level_config.at(level).first;
  int num_ter =
      (level == -1)
          ? (params.level_config.at(0).second - params.additional_base.second)
          : params.level_config.at(level).second;
  int num_q_primes = num_main + num_ter;
  grp.num_q_primes = num_q_primes;

  int max_ter = params.ter_primes.size();
  int padded_num_q_primes = num_main + max_ter;
  if (level == -1) padded_num_q_primes = num_q_primes;
  int num_pad = padded_num_q_primes - num_q_primes;

  int beta = (level == -1) ? 1 : ((num_main + max_ter + num_aux - 1) / num_aux);
  grp.beta = beta;

  std::vector<Word> level_primes = GetLevelPrimes(params, level, num_aux);

  grp.mod_up1.resize(num_q_primes);
  grp.digits.resize(beta);

  for (int i = 0; i < beta; i++) {
    int src_start = i * num_aux;
    int src_end = std::min((i + 1) * num_aux, padded_num_q_primes);
    if (src_end <= num_pad) continue;

    src_start = std::max(0, src_start - num_pad);
    src_end = src_end - num_pad;

    std::vector<Word> src_primes(level_primes.begin() + src_start,
                                 level_primes.begin() + src_end);
    std::vector<Word> dst_primes(level_primes.begin(),
                                 level_primes.begin() + src_start);
    dst_primes.insert(dst_primes.end(), level_primes.begin() + src_end,
                      level_primes.end());

    std::vector<Word> mod_up1_tmp;
    std::vector<SignedWord> mod_up2_tmp;
    PopulateModSwitchConstants(mod_up1_tmp, mod_up2_tmp, src_primes, dst_primes,
                               params.degree, 0, 0);

    for (size_t k = 0; k < mod_up1_tmp.size(); ++k) {
      grp.mod_up1[src_start + k] = mod_up1_tmp[k];
    }

    grp.digits[i].prime_index_start = src_start;
    grp.digits[i].prime_index_end = src_end;
    grp.digits[i].src_len = src_primes.size();
    grp.digits[i].dst_len = dst_primes.size();
    grp.digits[i].table = std::move(mod_up2_tmp);
  }

  return grp;
}

template <unsigned W>
ModDownTableGroup<W> ModSwitchTables<W>::GenerateModDown(
    const Params<W>& params, int level) {
  ModDownTableGroup<W> grp;
  grp.level = level;

  int num_aux = (level == -1) ? (params.level_config.at(0).first +
                                 params.level_config.at(0).second -
                                 params.additional_base.first -
                                 params.additional_base.second)
                              : params.alpha;
  grp.num_aux = num_aux;

  int num_main =
      (level == -1)
          ? (params.level_config.at(0).first - params.additional_base.first)
          : params.level_config.at(level).first;
  int num_ter =
      (level == -1)
          ? (params.level_config.at(0).second - params.additional_base.second)
          : params.level_config.at(level).second;
  int num_q_primes = num_main + num_ter;
  grp.num_q_primes = num_q_primes;
  grp.src_len = num_aux;
  grp.dst_len = num_q_primes;

  std::vector<Word> level_primes = GetLevelPrimes(params, level, num_aux);
  std::vector<Word> q_primes(level_primes.begin(),
                             level_primes.begin() + num_q_primes);
  std::vector<Word> p_primes(level_primes.begin() + num_q_primes,
                             level_primes.end());

  PopulateModSwitchConstants(grp.mod_down1, grp.mod_down2, p_primes, q_primes,
                             params.degree, 0, 0);

  std::vector<Word> dummy;
  PopulateModDownEpilogueConstants(grp.inv_prime_prod, dummy, p_primes,
                                   q_primes, 0, 0);

  return grp;
}

template <unsigned W>
RescaleTableGroup<W> ModSwitchTables<W>::GenerateRescale(
    const Params<W>& params, int level) {
  assert(level > 0 && "Rescale only supported for level > 0");
  RescaleTableGroup<W> grp;
  grp.level = level;

  int num_aux = params.alpha;
  std::vector<Word> level_primes = GetLevelPrimes(params, level, num_aux);
  int num_main = params.level_config.at(level).first;
  int num_ter = params.level_config.at(level).second;
  int num_q_primes = num_main + num_ter;

  int next_num_main = params.level_config.at(level - 1).first;
  int next_num_ter = params.level_config.at(level - 1).second;
  int next_num_q_primes = next_num_main + next_num_ter;
  std::vector<Word> next_level_primes =
      GetLevelPrimes(params, level - 1, num_aux);

  int main_diff = num_main - next_num_main;
  int ter_diff = num_ter - next_num_ter;

  std::vector<Word> rescale_dst_primes(
      next_level_primes.begin(), next_level_primes.begin() + next_num_q_primes);
  std::vector<Word> rescale_src_primes;

  if (ter_diff > 0) {
    rescale_src_primes.insert(rescale_src_primes.end(), level_primes.begin(),
                              level_primes.begin() + ter_diff);
    grp.pad_start = 0;
    grp.pad_end = num_q_primes - ter_diff;
    grp.restore_start = next_num_q_primes + main_diff;
    grp.restore_end = next_num_q_primes;
  } else if (main_diff > 0) {
    rescale_src_primes.insert(rescale_src_primes.end(),
                              level_primes.begin() + num_q_primes - main_diff,
                              level_primes.begin() + num_q_primes);
    grp.pad_start = -ter_diff;
    grp.pad_end = num_q_primes - main_diff - ter_diff;
    grp.restore_start = 0;
    grp.restore_end = -ter_diff;
  } else {
    assert(false && "Invalid rescale amount");
  }

  grp.src_len = rescale_src_primes.size();
  grp.dst_len = rescale_dst_primes.size();

  PopulateModSwitchConstants(grp.rescale1, grp.rescale2, rescale_src_primes,
                             rescale_dst_primes, params.degree,
                             grp.restore_start, grp.restore_end);
  PopulateModDownEpilogueConstants(grp.inv_prime_prod, grp.src2_padding,
                                   rescale_src_primes, rescale_dst_primes,
                                   grp.restore_start, grp.restore_end);

  return grp;
}

template <unsigned W>
ModDownRescaleTableGroup<W> ModSwitchTables<W>::GenerateModDownRescale(
    const Params<W>& params, int level) {
  assert(level > 0 && "ModDownAndRescale only supported for level > 0");
  ModDownRescaleTableGroup<W> grp;
  grp.level = level;

  int num_aux = params.alpha;
  std::vector<Word> level_primes = GetLevelPrimes(params, level, num_aux);
  int num_main = params.level_config.at(level).first;
  int num_ter = params.level_config.at(level).second;
  int num_q_primes = num_main + num_ter;

  std::vector<Word> p_primes(level_primes.begin() + num_q_primes,
                             level_primes.end());

  int next_num_main = params.level_config.at(level - 1).first;
  int next_num_ter = params.level_config.at(level - 1).second;
  int next_num_q_primes = next_num_main + next_num_ter;
  std::vector<Word> next_level_primes =
      GetLevelPrimes(params, level - 1, num_aux);

  int main_diff = num_main - next_num_main;
  int ter_diff = num_ter - next_num_ter;

  std::vector<Word> rescale_dst_primes(
      next_level_primes.begin(), next_level_primes.begin() + next_num_q_primes);
  std::vector<Word> rescale_src_primes;
  std::vector<Word> mod_down_rescale_src_primes;

  if (ter_diff > 0) {
    rescale_src_primes.insert(rescale_src_primes.end(), level_primes.begin(),
                              level_primes.begin() + ter_diff);
    mod_down_rescale_src_primes = rescale_src_primes;
    mod_down_rescale_src_primes.insert(mod_down_rescale_src_primes.end(),
                                       p_primes.begin(), p_primes.end());
    grp.pad_start = 0;
    grp.pad_end = num_q_primes - ter_diff;
    grp.restore_start = next_num_q_primes + main_diff;
    grp.restore_end = next_num_q_primes;
  } else if (main_diff > 0) {
    rescale_src_primes.insert(rescale_src_primes.end(),
                              level_primes.begin() + num_q_primes - main_diff,
                              level_primes.begin() + num_q_primes);
    mod_down_rescale_src_primes = rescale_src_primes;
    mod_down_rescale_src_primes.insert(mod_down_rescale_src_primes.end(),
                                       p_primes.begin(), p_primes.end());
    grp.pad_start = -ter_diff;
    grp.pad_end = num_q_primes - main_diff - ter_diff;
    grp.restore_start = 0;
    grp.restore_end = -ter_diff;
  } else {
    assert(false && "Invalid rescale amount");
  }

  grp.src_len = mod_down_rescale_src_primes.size();
  grp.dst_len = rescale_dst_primes.size();

  PopulateModSwitchConstants(grp.mod_down_rescale1, grp.mod_down_rescale2,
                             mod_down_rescale_src_primes, rescale_dst_primes,
                             params.degree, grp.restore_start, grp.restore_end);
  PopulateModDownEpilogueConstants(
      grp.inv_prime_prod, grp.src2_padding, mod_down_rescale_src_primes,
      rescale_dst_primes, grp.restore_start, grp.restore_end);

  return grp;
}

template struct PseudoModUpTableGroup<32>;
template struct PseudoModUpTableGroup<64>;
template struct ModUpTableGroup<32>;
template struct ModUpTableGroup<64>;
template struct ModDownTableGroup<32>;
template struct ModDownTableGroup<64>;
template struct RescaleTableGroup<32>;
template struct RescaleTableGroup<64>;
template struct ModDownRescaleTableGroup<32>;
template struct ModDownRescaleTableGroup<64>;

template class ModSwitchTables<32>;
template class ModSwitchTables<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
