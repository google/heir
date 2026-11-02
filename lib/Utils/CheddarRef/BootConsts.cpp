#include "lib/Utils/CheddarRef/BootConsts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "lib/Utils/CheddarRef/Encoder.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir::heir::cheddar_ref {
namespace {

constexpr double kZeroCoeffThreshold = 1e-9;

const std::vector<double>& GetModCoefficients() {
  static const std::vector<double> kModCoefficients = {
      0.12517186708929745802,    0.0, 0.2894364973331168731,      0.0,
      0.36272381596524499154,    0.0, 0.3011054704600794278,      0.0,
      -0.10550875667295944105,   0.0, -0.43588877795190139706,    0.0,
      0.37482647434055190702,    0.0, -0.14821069913569220404,    0.0,
      0.03665437786710548091,    0.0, -0.0063882548960017121343,  0.0,
      0.00083684232451067872756, 0.0, -8.6443599931576702305e-05, 0.0,
      7.0966437900548814324e-06, 0.0, -5.228015817181348194e-07,  0.0,
      2.2714690137973883081e-08, 0.0, -2.3761936068138980797e-09};
  return kModCoefficients;
}

inline int Log2Ceil(int n) {
  int res = 0;
  while ((1 << res) < n) ++res;
  return res;
}

inline int Log2Floor(int n) {
  int res = 0;
  while ((1 << (res + 1)) <= n) ++res;
  return res;
}

inline int DivCeil(int a, int b) { return (a + b - 1) / b; }

inline int Log2Scale(double scale) {
  return static_cast<int>(std::log2(scale) + 0.5);
}

void RemoveZero(std::vector<double>& coefficients) {
  while (!coefficients.empty() &&
         std::abs(coefficients.back()) < kZeroCoeffThreshold) {
    coefficients.pop_back();
  }
  for (auto& c : coefficients) {
    if (std::abs(c) < kZeroCoeffThreshold) {
      c = 0.0;
    }
  }
}

template <unsigned W>
AXYPBZConsts<W> CreateAXYPBZ(const Params<W>& params, const Encoder<W>& encoder,
                             int a, double b, int x_level, double x_scale,
                             int y_level, double y_scale, bool has_z,
                             int z_level, double z_scale) {
  AXYPBZConsts<W> res;
  res.x_level = x_level;
  res.x_scale = x_scale;
  res.y_level = y_level;
  res.y_scale = y_scale;
  res.has_z = has_z;
  res.z_level = z_level;
  res.z_scale = z_scale;

  double working_scale = x_scale * y_scale;
  int working_level = has_z ? std::min({x_level, y_level, z_level})
                            : std::min(x_level, y_level);

  if (b == 0.0) {
    res.has_b = false;
  } else {
    res.has_b = true;
    double b_scale = has_z ? (working_scale / z_scale) : working_scale;
    res.b_cx = encoder.EncodeConstant(working_level, b_scale, b);
  }

  if (a == 1) {
    res.has_a = false;
  } else {
    res.has_a = true;
    res.a_cx =
        encoder.EncodeConstant(working_level, 1.0, static_cast<double>(a));
  }

  res.final_level = working_level - 1;
  res.final_scale = working_scale / encoder.GetRescalePrimeProd(working_level);
  return res;
}

inline EvalPolyType DetermineType(std::vector<double>& coefficients) {
  RemoveZero(coefficients);
  int degree = coefficients.size() - 1;
  bool odd_flag = true;
  bool even_flag = true;
  for (int i = 0; i <= degree; i += 2) {
    if (coefficients[i] != 0.0) odd_flag = false;
  }
  for (int i = 1; i <= degree; i += 2) {
    if (coefficients[i] != 0.0) even_flag = false;
  }
  if (even_flag) return EvalPolyType::kEven;
  if (odd_flag) return EvalPolyType::kOdd;
  return EvalPolyType::kNormal;
}

}  // namespace

// =============================================================================
// RefBasisMap Implementation
// =============================================================================

template <unsigned W>
RefBasisMap<W>::RefBasisMap(const Params<W>& params, const Encoder<W>& encoder,
                            int input_level, double input_scale,
                            EvalPolyType type)
    : params_(params),
      encoder_(encoder),
      input_level_(input_level),
      input_scale_(input_scale),
      type_(type) {}

template <unsigned W>
int RefBasisMap<W>::SplitBaseDegree(int base_degree) const {
  int split = 0;
  if ((base_degree & (base_degree - 1)) == 0) {
    split = base_degree / 2;
  } else if (type_ == EvalPolyType::kEven) {
    split = 1 << Log2Floor(base_degree);
  } else {
    split = (1 << Log2Floor(base_degree)) - 1;
  }
  return std::max(split, base_degree - split);
}

template <unsigned W>
bool RefBasisMap<W>::Exists(int base_degree) const {
  return (base_degree == 1) ||
         (basis_eval_.find(base_degree) != basis_eval_.end());
}

template <unsigned W>
std::pair<int, double> RefBasisMap<W>::GetBaseLevelAndScale(
    int base_degree) const {
  if (base_degree == 1) {
    return {input_level_, input_scale_};
  }
  const auto& eval = basis_eval_.at(base_degree);
  return {eval.final_level, eval.final_scale};
}

template <unsigned W>
bool RefBasisMap<W>::IsMultUnsafeCompatible(int level1, int level2) const {
  if (level1 == level2) return true;
  int min_l = std::min(level1, level2);
  int max_l = std::max(level1, level2);
  NPInfo min_np = encoder_.LevelToNP(min_l, 0);
  NPInfo max_np = encoder_.LevelToNP(max_l, 0);
  return min_np.num_main <= max_np.num_main && min_np.num_ter <= max_np.num_ter;
}

template <unsigned W>
void RefBasisMap<W>::AddBase(int base_degree) {
  int left_degree = SplitBaseDegree(base_degree);
  int right_degree = base_degree - left_degree;

  if (!Exists(left_degree)) AddBase(left_degree);
  if (!Exists(right_degree)) AddBase(right_degree);

  auto [left_level, left_scale] = GetBaseLevelAndScale(left_degree);
  auto [right_level, right_scale] = GetBaseLevelAndScale(right_degree);
  int working_level = std::min(left_level, right_level);

  while (!IsMultUnsafeCompatible(left_level, working_level)) {
    left_level -= 1;
  }
  while (!IsMultUnsafeCompatible(right_level, working_level)) {
    right_level -= 1;
  }

  int sub_degree = left_degree - right_degree;
  if (sub_degree == 0) {
    basis_eval_[base_degree] =
        CreateAXYPBZ<W>(params_, encoder_, 2, -1.0, left_level, left_scale,
                        right_level, right_scale, /*has_z=*/false, 0, 0.0);
  } else {
    if (!Exists(sub_degree)) AddBase(sub_degree);
    auto [sub_level, sub_scale] = GetBaseLevelAndScale(sub_degree);
    while (!IsMultUnsafeCompatible(sub_level, working_level)) {
      sub_level -= 1;
    }
    basis_eval_[base_degree] = CreateAXYPBZ<W>(
        params_, encoder_, 2, -1.0, left_level, left_scale, right_level,
        right_scale, /*has_z=*/true, sub_level, sub_scale);
  }
}

// =============================================================================
// RefEvalPolyNode Implementation
// =============================================================================

template <unsigned W>
RefEvalPolyNode<W>::RefEvalPolyNode(const std::vector<double>& coefficients,
                                    int level_margin, int baby_threshold)
    : coefficients_(coefficients) {
  int degree = coefficients.size() - 1;
  int level_consumed = Log2Ceil(degree) + 1;
  if (degree < baby_threshold && level_consumed <= level_margin) {
    for (int i = 0; i <= degree; ++i) {
      if (std::abs(coefficients[i]) > kZeroCoeffThreshold) {
        leaf_constants_[i] = {};
      }
    }
    return;
  }

  split_degree_ = 1 << Log2Floor(degree);
  std::vector<double> low_coeffs(coefficients.begin(),
                                 coefficients.begin() + split_degree_);
  std::vector<double> high_coeffs(coefficients.begin() + split_degree_,
                                  coefficients.end());
  for (size_t i = 1; i < high_coeffs.size(); ++i) {
    low_coeffs[split_degree_ - i] -= high_coeffs[i];
    high_coeffs[i] *= 2;
  }
  RemoveZero(low_coeffs);
  RemoveZero(high_coeffs);

  is_low_zero_ = low_coeffs.empty();
  is_low_constant_ = (low_coeffs.size() <= 1);
  is_high_constant_ = (high_coeffs.size() == 1);

  if (!is_low_constant_) {
    low_ = std::make_shared<RefEvalPolyNode<W>>(low_coeffs, level_margin,
                                                baby_threshold);
  }
  if (!is_high_constant_) {
    high_ = std::make_shared<RefEvalPolyNode<W>>(high_coeffs, level_margin - 1,
                                                 baby_threshold);
  }
}

template <unsigned W>
void RefEvalPolyNode<W>::CheckRequiredBasis(
    std::set<int>& required_bases) const {
  for (const auto& [deg, _] : leaf_constants_) {
    required_bases.insert(deg);
  }
  if (low_ != nullptr) low_->CheckRequiredBasis(required_bases);
  if (high_ != nullptr) high_->CheckRequiredBasis(required_bases);
  required_bases.insert(split_degree_);
}

template <unsigned W>
void RefEvalPolyNode<W>::Compile(const Encoder<W>& encoder,
                                 const RefBasisMap<W>& basis_map,
                                 int target_level, double target_scale,
                                 bool do_rescale) {
  target_level_ = target_level;
  target_scale_ = target_scale;
  do_rescale_ = do_rescale;

  int working_level = target_level;
  double working_scale = target_scale;
  if (do_rescale) {
    working_level = target_level + 1;
    working_scale = target_scale * encoder.GetRescalePrimeProd(working_level);
  }

  if (is_low_constant_ && (!is_low_zero_)) {
    low_constant_ =
        encoder.EncodeConstant(working_level, working_scale, coefficients_[0]);
  } else if (low_ != nullptr) {
    low_->Compile(encoder, basis_map, working_level, working_scale, false);
  }

  if (is_high_constant_ || high_ != nullptr) {
    auto [_, split_scale] = basis_map.GetBaseLevelAndScale(split_degree_);
    double high_scale = working_scale / split_scale;
    if (is_high_constant_) {
      high_constant_ = encoder.EncodeConstant(working_level, high_scale,
                                              coefficients_[split_degree_]);
    } else {
      high_->Compile(encoder, basis_map, working_level, high_scale, true);
    }
  }

  for (auto& [base_degree, constant] : leaf_constants_) {
    double value = coefficients_[base_degree];
    double scale = working_scale;
    if (base_degree != 0) {
      auto [_, base_scale] = basis_map.GetBaseLevelAndScale(base_degree);
      scale /= base_scale;
    }
    constant = encoder.EncodeConstant(working_level, scale, value);
  }
}

template <unsigned W>
void RefEvalPolyNode<W>::CollectLimbs(std::vector<Word>& limbs) {
  if (is_low_constant_ && (!is_low_zero_)) {
    low_const_offset_ = static_cast<int64_t>(limbs.size());
    limbs.insert(limbs.end(), low_constant_.begin(), low_constant_.end());
  }
  if (low_ != nullptr) low_->CollectLimbs(limbs);
  if (is_high_constant_) {
    high_const_offset_ = static_cast<int64_t>(limbs.size());
    limbs.insert(limbs.end(), high_constant_.begin(), high_constant_.end());
  } else if (high_ != nullptr) {
    high_->CollectLimbs(limbs);
  }
  for (const auto& [deg, constant] : leaf_constants_) {
    leaf_const_offsets_[deg] = static_cast<int64_t>(limbs.size());
    limbs.insert(limbs.end(), constant.begin(), constant.end());
  }
}

// =============================================================================
// BootConsts Implementation
// =============================================================================

template <unsigned W>
BootConsts<W>::BootConsts(const Params<W>& params, const Tables<W>& tables,
                          const Encoder<W>& encoder, int num_cts_levels,
                          int num_stc_levels, int log_message_ratio,
                          int num_double_angle, int initial_K)
    : params_(params),
      tables_(tables),
      encoder_(encoder),
      num_cts_levels_(num_cts_levels),
      num_stc_levels_(num_stc_levels),
      log_message_ratio_(log_message_ratio),
      num_double_angle_(num_double_angle),
      initial_K_(initial_K) {
  InitScalarsAndContextConsts();
  InitEvalMod();
}

template <unsigned W>
void BootConsts<W>::InitScalarsAndContextConsts() {
  int max_level = params_.level_config.size() - 1;
  scalars_.max_level = max_level;
  scalars_.initial_K = initial_K_;
  scalars_.num_double_angle = num_double_angle_;

  int num_eval_mod_levels =
      Log2Ceil(GetModCoefficients().size()) + num_double_angle_;
  scalars_.cts_start_level = max_level;
  scalars_.eval_mod_start_level = max_level - num_cts_levels_;
  scalars_.stc_start_level =
      scalars_.eval_mod_start_level - num_eval_mod_levels;
  scalars_.end_level = scalars_.stc_start_level - num_stc_levels_;

  // Short base (level -1) NP
  int short_main =
      params_.level_config.at(0).first - params_.additional_base.first;
  int short_ter =
      params_.level_config.at(0).second - params_.additional_base.second;

  std::vector<Word> q0;
  int prime_offset = params_.ter_primes.size() - short_ter;
  for (int i = 0; i < short_ter; ++i) {
    q0.push_back(params_.all_primes[prime_offset + i]);
  }
  for (int i = 0; i < short_main; ++i) {
    q0.push_back(params_.all_primes[params_.ter_primes.size() + i]);
  }

  double q0_prod = 1.0;
  for (Word q : q0) {
    q0_prod *= static_cast<double>(q);
  }
  int log_q0_prod = Log2Scale(q0_prod);

  double base_scale =
      static_cast<double>(UINT64_C(1) << params_.scale_info.log_default_scale);
  int log_level_zero_scale = Log2Scale(base_scale);
  scalars_.log_scaleup =
      (log_q0_prod - log_level_zero_scale) - log_message_ratio_;

  double level_zero_scale =
      base_scale * static_cast<double>(1 << scalars_.log_scaleup);

  int log_eval_mod_start_scale =
      Log2Scale(encoder_.GetRescalePrimeProd(scalars_.eval_mod_start_level));
  int actual_K = (1 << num_double_angle_) * initial_K_;

  double eval_mod_end_scale =
      static_cast<double>(UINT64_C(1) << log_eval_mod_start_scale);
  for (int i = 0; i < num_eval_mod_levels; ++i) {
    eval_mod_end_scale =
        eval_mod_end_scale * eval_mod_end_scale /
        encoder_.GetRescalePrimeProd(scalars_.eval_mod_start_level - i);
  }

  scalars_.cts_const =
      static_cast<double>(UINT64_C(1)
                          << (log_eval_mod_start_scale - params_.log_n)) /
      (q0_prod * actual_K);
  scalars_.stc_const = (q0_prod * encoder_.GetScale(scalars_.end_level)) /
                       (eval_mod_end_scale * level_zero_scale);

  // Scaleup constant: level -1, scale 1.0, value (1 << log_scaleup_)
  scaleup_const_ = encoder_.EncodeConstant(
      -1, 1.0, static_cast<double>(1 << scalars_.log_scaleup));

  // ModMax INTT constant: intt_const[i] = InvMod(prod_{j!=i} q0[j] * degree,
  // q0[i])
  int num_base = q0.size();
  mod_max_intt_const_.resize(num_base, 1);
  for (int i = 0; i < num_base; ++i) {
    Word mod_prime = q0[i];
    for (int j = 0; j < num_base; ++j) {
      if (i != j) {
        mod_max_intt_const_[i] = Tables<W>::MultMod(
            mod_max_intt_const_[i], q0[j] % mod_prime, mod_prime);
      }
    }
    mod_max_intt_const_[i] = Tables<W>::MultMod(
        mod_max_intt_const_[i], static_cast<Word>(params_.degree % mod_prime),
        mod_prime);
    mod_max_intt_const_[i] =
        Tables<W>::InvMod(mod_max_intt_const_[i], mod_prime);
  }

  // Level down constants (levels 1 .. default_encryption_level)
  int default_enc_level = scalars_.stc_start_level;
  level_down_consts_.clear();
  level_down_consts_.resize(default_enc_level + 1);
  for (int i = default_enc_level; i > 0; --i) {
    double scale = encoder_.GetScale(i);
    level_down_consts_[i] = encoder_.EncodeConstant(i, scale, 1.0);
  }

  // Imaginary unit constants: psi_rev_mont[l*N+1] for each prime in all_primes
  imaginary_unit_consts_.clear();
  imaginary_unit_consts_.resize(params_.all_primes.size());
  for (size_t l = 0; l < params_.all_primes.size(); ++l) {
    if (tables_.psi_rev_mont.size() >= (l + 1) * params_.degree) {
      imaginary_unit_consts_[l] = tables_.psi_rev_mont[l * params_.degree + 1];
    } else {
      imaginary_unit_consts_[l] = 0;
    }
  }
}

template <unsigned W>
void BootConsts<W>::InitEvalMod() {
  int start_level = scalars_.eval_mod_start_level;
  int first_log_scale = Log2Scale(encoder_.GetRescalePrimeProd(start_level));
  double start_scale = static_cast<double>(UINT64_C(1) << first_log_scale);

  int actual_K = (1 << num_double_angle_) * initial_K_;
  evalmod_initial_const_ = encoder_.EncodeConstant(
      start_level, start_scale, -0.25 / static_cast<double>(actual_K));

  const auto& mod_coefficients = GetModCoefficients();
  int mod_levels = Log2Ceil(mod_coefficients.size());

  double target_scale = start_scale;
  for (int i = 0; i < mod_levels; ++i) {
    target_scale = target_scale * target_scale /
                   encoder_.GetRescalePrimeProd(start_level - i);
  }

  // Compile EvalPolyNode tree
  int level_consumption = Log2Ceil(mod_coefficients.size());
  int baby_threshold = 1 << DivCeil(level_consumption, 2);
  auto tree_root = std::make_shared<RefEvalPolyNode<W>>(
      mod_coefficients, level_consumption, baby_threshold);

  std::set<int> required_base_degrees;
  tree_root->CheckRequiredBasis(required_base_degrees);

  auto mod_coeffs_copy = mod_coefficients;
  EvalPolyType poly_type = DetermineType(mod_coeffs_copy);
  basis_map_ = std::make_shared<RefBasisMap<W>>(params_, encoder_, start_level,
                                                start_scale, poly_type);
  for (int deg : required_base_degrees) {
    if (deg == 0 || deg == 1) continue;
    basis_map_->AddBase(deg);
  }

  int target_level = start_level - level_consumption;
  tree_root->Compile(encoder_, *basis_map_, target_level, target_scale);

  evalpoly_consts_.clear();
  tree_root->CollectLimbs(evalpoly_consts_);
  evalpoly_root_ = tree_root;

  // Double angle steps
  double_angle_.clear();
  double sqrt2pi =
      std::pow(0.5 / M_PI, 1.0 / static_cast<double>(1 << num_double_angle_));
  for (int i = 0; i < num_double_angle_; ++i) {
    sqrt2pi *= sqrt2pi;
    int double_angle_level = start_level - mod_levels - i;
    double_angle_.push_back(CreateAXYPBZ<W>(
        params_, encoder_, 2, -sqrt2pi, double_angle_level, target_scale,
        double_angle_level, target_scale, /*has_z=*/false, 0, 0.0));
    target_scale = target_scale * target_scale /
                   encoder_.GetRescalePrimeProd(double_angle_level);
  }
}

template <unsigned W>
StripedMat BootConsts<W>::MultStriped(const StripedMat& a, const StripedMat& b,
                                      int width) {
  StripedMat c;
  for (const auto& [i, diag_a] : a) {
    for (const auto& [j, diag_b] : b) {
      int dest_idx = (i + j) % width;
      if (c.find(dest_idx) == c.end()) {
        c[dest_idx] = std::vector<Complex>(width, Complex(0.0, 0.0));
      }
      for (int k = 0; k < width; ++k) {
        c[dest_idx][k] += (diag_a[k] * diag_b[(k + i) % width]);
      }
    }
  }
  return c;
}

template <unsigned W>
StripedMat BootConsts<W>::ScaleStriped(const StripedMat& a, Complex scalar) {
  StripedMat c;
  for (const auto& [i, diag] : a) {
    c[i].resize(diag.size());
    for (size_t j = 0; j < diag.size(); ++j) {
      c[i][j] = diag[j] * scalar;
    }
  }
  return c;
}

template <unsigned W>
std::vector<StripedMat> BootConsts<W>::GenerateFFTStages(int num_slots) const {
  int num_stages = Log2Ceil(num_slots);
  int M = params_.degree * 2;
  std::vector<StripedMat> fft_stages(num_stages);

  for (int i = 0; i < num_stages; ++i) {
    int stride = 1 << i;
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M / st8;

    auto& target = fft_stages[i];
    target[0] = std::vector<Complex>(num_slots, Complex(0.0, 0.0));
    target[stride] = std::vector<Complex>(num_slots, Complex(0.0, 0.0));
    if (i != num_stages - 1) {
      target[num_slots - stride] =
          std::vector<Complex>(num_slots, Complex(0.0, 0.0));
    }

    auto& diag_0 = target[0];
    auto& diag_plus = target[stride];
    auto& diag_minus = target[num_slots - stride];

    for (int j = 0; j < stride; ++j) {
      int twiddle_idx = (encoder_.GetGaloisFactor(j) % st8) * gap;
      Complex twiddle = encoder_.GetTwiddleFactor(twiddle_idx);

      diag_0[j] = 1.0;
      diag_plus[j] = twiddle;
      diag_minus[j + stride] = 1.0;
      diag_0[j + stride] = -twiddle;
    }

    int num_double = Log2Ceil(num_slots / stride_group_size);
    for (int r = 0; r < num_double; ++r) {
      size_t copy_len = stride_group_size * (1 << r);
      std::copy_n(diag_0.begin(), copy_len, diag_0.begin() + copy_len);
      std::copy_n(diag_plus.begin(), copy_len, diag_plus.begin() + copy_len);
      if (i != num_stages - 1) {
        std::copy_n(diag_minus.begin(), copy_len,
                    diag_minus.begin() + copy_len);
      }
    }
  }
  return fft_stages;
}

template <unsigned W>
std::vector<StripedMat> BootConsts<W>::GenerateIFFTStages(int num_slots) const {
  int num_stages = Log2Ceil(num_slots);
  int M = params_.degree * 2;
  std::vector<StripedMat> ifft_stages(num_stages);

  for (int i = 0; i < num_stages; ++i) {
    int stride = 1 << i;
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M / st8;

    auto& target = ifft_stages[num_stages - i - 1];
    target[0] = std::vector<Complex>(num_slots, Complex(0.0, 0.0));
    target[stride] = std::vector<Complex>(num_slots, Complex(0.0, 0.0));
    if (i != num_stages - 1) {
      target[num_slots - stride] =
          std::vector<Complex>(num_slots, Complex(0.0, 0.0));
    }

    auto& diag_0 = target[0];
    auto& diag_plus = target[stride];
    auto& diag_minus = target[num_slots - stride];

    for (int j = 0; j < stride; ++j) {
      int twiddle_idx = (st8 - (encoder_.GetGaloisFactor(j) % st8)) * gap;
      Complex twiddle = encoder_.GetTwiddleFactor(twiddle_idx);

      diag_0[j] = 1.0;
      diag_plus[j] = 1.0;
      diag_minus[j + stride] = twiddle;
      diag_0[j + stride] = -twiddle;
    }

    int num_double = Log2Ceil(num_slots / stride_group_size);
    for (int r = 0; r < num_double; ++r) {
      size_t copy_len = stride_group_size * (1 << r);
      std::copy_n(diag_0.begin(), copy_len, diag_0.begin() + copy_len);
      std::copy_n(diag_plus.begin(), copy_len, diag_plus.begin() + copy_len);
      if (i != num_stages - 1) {
        std::copy_n(diag_minus.begin(), copy_len,
                    diag_minus.begin() + copy_len);
      }
    }
  }
  return ifft_stages;
}

template <unsigned W>
std::vector<StripedMat> BootConsts<W>::GenerateCtSPhaseMatrices(
    int num_slots) const {
  auto ifft_stages = GenerateIFFTStages(num_slots);
  int log_num_slots = Log2Ceil(num_slots);
  int cts_stages_left = log_num_slots;
  int cts_stages_cumul = 0;
  double cts_const_div = std::pow(scalars_.cts_const, 1.0 / num_cts_levels_);
  bool full_slot = (num_slots == static_cast<int>(params_.degree / 2));

  std::vector<StripedMat> phases;
  phases.reserve(num_cts_levels_);

  for (int i = 0; i < num_cts_levels_; ++i) {
    int num_stages = (i == 0) ? DivCeil(cts_stages_left, num_cts_levels_)
                              : (cts_stages_left / (num_cts_levels_ - i));
    cts_stages_left -= num_stages;

    StripedMat phase_matrix = ifft_stages[cts_stages_cumul];
    for (int j = cts_stages_cumul + 1; j < cts_stages_cumul + num_stages; ++j) {
      phase_matrix = MultStriped(ifft_stages[j], phase_matrix, num_slots);
    }

    if (i == num_cts_levels_ - 1 && !full_slot) {
      StripedMat extended;
      for (const auto& [idx, diag] : phase_matrix) {
        int dst_idx = idx;
        if (idx >= num_slots / 2) dst_idx += num_slots;
        extended[dst_idx] =
            std::vector<Complex>(num_slots * 2, Complex(0.0, 0.0));
        for (int k = 0; k < num_slots; ++k) {
          extended[dst_idx][k] = diag[k];
          extended[dst_idx][k + num_slots] = diag[k] * Complex(0.0, -1.0);
        }
      }
      phase_matrix = extended;
    }

    phase_matrix = ScaleStriped(phase_matrix, cts_const_div);
    phases.push_back(phase_matrix);
    cts_stages_cumul += num_stages;
  }
  return phases;
}

template <unsigned W>
std::vector<StripedMat> BootConsts<W>::GenerateStCPhaseMatrices(
    int num_slots) const {
  auto fft_stages = GenerateFFTStages(num_slots);
  int log_num_slots = Log2Ceil(num_slots);
  int stc_stages_left = log_num_slots;
  int stc_stages_cumul = 0;
  double stc_const_div = std::pow(scalars_.stc_const, 1.0 / num_stc_levels_);
  bool full_slot = (num_slots == static_cast<int>(params_.degree / 2));

  std::vector<StripedMat> phases;
  phases.reserve(num_stc_levels_);

  for (int i = 0; i < num_stc_levels_; ++i) {
    int num_stages = stc_stages_left / (num_stc_levels_ - i);
    stc_stages_left -= num_stages;

    StripedMat phase_matrix = fft_stages[stc_stages_cumul];
    for (int j = stc_stages_cumul + 1; j < stc_stages_cumul + num_stages; ++j) {
      phase_matrix = MultStriped(fft_stages[j], phase_matrix, num_slots);
    }

    if (i == 0 && !full_slot) {
      StripedMat extended;
      for (const auto& [idx, diag] : phase_matrix) {
        int dst_idx = idx;
        if (idx >= num_slots / 2) dst_idx += num_slots;
        extended[dst_idx] =
            std::vector<Complex>(num_slots * 2, Complex(0.0, 0.0));
        for (int k = 0; k < num_slots; ++k) {
          extended[dst_idx][k] = diag[k];
          extended[dst_idx][k + num_slots] = diag[k] * Complex(0.0, 1.0);
        }
      }
      phase_matrix = extended;
    }

    phase_matrix = ScaleStriped(phase_matrix, stc_const_div);
    phases.push_back(phase_matrix);
    stc_stages_cumul += num_stages;
  }
  return phases;
}

template <unsigned W>
std::pair<int, int> BootConsts<W>::BSGSSplit(int num_diag) {
  if (num_diag <= 4) {
    return {num_diag, 1};
  }
  int bs = 0;
  switch (num_diag) {
    case 7:
    case 8:
    case 15:
    case 16:
      bs = 4;
      break;
    case 31:
    case 32:
    case 63:
    case 64:
      bs = 8;
      break;
    default:
      bs = 1 << DivCeil(Log2Ceil(num_diag), 2);
      break;
  }
  int gs = DivCeil(num_diag, bs);
  return {bs, gs};
}

template <unsigned W>
typename BootConsts<W>::PlainHoistMap BootConsts<W>::ConstructPlainHoistMap(
    const StripedMat& matrix, int bs, int gs, int pre_rotation,
    int additional_pt_rot) {
  if (matrix.empty()) return {};
  int height = matrix.begin()->second.size();
  int width = height;
  int gcd_rot = 0;
  for (const auto& [i, _] : matrix) {
    int rot = (i - pre_rotation) % width;
    if (rot < 0) rot += width;
    if (gcd_rot == 0 && rot == 0) {
      gcd_rot = rot;
    } else {
      gcd_rot = std::gcd(gcd_rot, rot);
    }
  }

  int gs_stride = gcd_rot * bs;
  PlainHoistMap hoist_map;
  for (const auto& [i, diag] : matrix) {
    int rot = (i - pre_rotation) % width;
    if (rot < 0) rot += width;
    int bs_rot = rot % gs_stride;
    int gs_rot = rot - bs_rot;
    if (hoist_map.find(gs_rot) == hoist_map.end()) {
      hoist_map.try_emplace(gs_rot, std::map<int, std::vector<Complex>>());
    }
    hoist_map[gs_rot][bs_rot] = std::vector<Complex>(height, Complex(0.0, 0.0));
    int offset = (gs_rot - additional_pt_rot) % height;
    if (offset < 0) offset += height;
    for (int j = 0; j < height; ++j) {
      hoist_map[gs_rot][bs_rot][(j + offset) % height] = diag[j];
    }
  }
  return hoist_map;
}

template <unsigned W>
std::vector<typename BootConsts<W>::Word> BootConsts<W>::EncodeDiagonal(
    const std::vector<Complex>& diag, int pt_level, double pt_scale,
    int num_aux) const {
  return encoder_.Encode(pt_level, pt_scale, diag, num_aux);
}

template <unsigned W>
LinearTransformPlan BootConsts<W>::GetLinearTransformPlan(int num_slots) const {
  LinearTransformPlan plan;
  int log_num_slots = Log2Ceil(num_slots);

  int64_t current_slab_words = 0;

  // 1. CtS Phases
  auto cts_phase_matrices = GenerateCtSPhaseMatrices(num_slots);
  int cts_stages_left = log_num_slots;
  for (int i = 0; i < num_cts_levels_; ++i) {
    int num_stages = (i == 0) ? DivCeil(cts_stages_left, num_cts_levels_)
                              : (cts_stages_left / (num_cts_levels_ - i));
    cts_stages_left -= num_stages;

    int num_eff_diag = cts_phase_matrices[i].size();
    if (i == num_cts_levels_ - 1) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    int pre_rotation = 0;
    int additional_pt_rot = -(1 << cts_stages_left);
    if (i == 0) {
      pre_rotation = (1 << cts_stages_left);
    } else if (i == num_cts_levels_ - 1) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << cts_stages_left);
    }

    int pt_level = scalars_.cts_start_level - i;
    double pt_scale = encoder_.GetRescalePrimeProd(pt_level);

    auto hoist_map = ConstructPlainHoistMap(cts_phase_matrices[i], bs, gs,
                                            pre_rotation, additional_pt_rot);

    bool bs_gs_swapped =
        (hoist_map.size() == 1 && hoist_map.begin()->first == 0);

    LinearTransformPhaseSpec spec;
    spec.pt_level = pt_level;
    spec.pt_scale = pt_scale;
    spec.bs = bs;
    spec.gs = gs;
    spec.bs_gs_swapped = bs_gs_swapped;
    spec.pre_rotation = pre_rotation;
    spec.additional_pt_rot = additional_pt_rot;
    spec.num_limbs = (pt_level + 1) + params_.alpha;

    if (bs_gs_swapped) {
      spec.bs_indices = {0};
      for (const auto& [bs_idx, _] : hoist_map.begin()->second) {
        spec.gs_indices.push_back(bs_idx);
      }
      std::sort(spec.gs_indices.begin(), spec.gs_indices.end());
      spec.pt_slab_word_offset.resize(spec.gs_indices.size(),
                                      std::vector<int64_t>(1, -1));
      for (size_t g = 0; g < spec.gs_indices.size(); ++g) {
        spec.pt_slab_word_offset[g][0] = current_slab_words;
        current_slab_words +=
            static_cast<int64_t>(spec.num_limbs) * params_.degree;
      }
    } else {
      std::set<int> bs_set;
      for (const auto& [gs_idx, bs_map] : hoist_map) {
        spec.gs_indices.push_back(gs_idx);
        for (const auto& [bs_idx, _] : bs_map) {
          bs_set.insert(bs_idx);
        }
      }
      std::sort(spec.gs_indices.begin(), spec.gs_indices.end());
      spec.bs_indices.assign(bs_set.begin(), bs_set.end());
      spec.pt_slab_word_offset.resize(
          spec.gs_indices.size(),
          std::vector<int64_t>(spec.bs_indices.size(), -1));
      for (size_t g = 0; g < spec.gs_indices.size(); ++g) {
        int gs_idx = spec.gs_indices[g];
        for (size_t b = 0; b < spec.bs_indices.size(); ++b) {
          int bs_idx = spec.bs_indices[b];
          if (hoist_map.count(gs_idx) && hoist_map.at(gs_idx).count(bs_idx)) {
            spec.pt_slab_word_offset[g][b] = current_slab_words;
            current_slab_words +=
                static_cast<int64_t>(spec.num_limbs) * params_.degree;
          }
        }
      }
    }
    plan.cts.push_back(std::move(spec));
  }

  // 2. StC Phases
  auto stc_phase_matrices = GenerateStCPhaseMatrices(num_slots);
  int stc_stages_left = log_num_slots;
  int stc_stages_cumul = 0;
  for (int i = 0; i < num_stc_levels_; ++i) {
    int num_stages = stc_stages_left / (num_stc_levels_ - i);
    stc_stages_left -= num_stages;

    int num_eff_diag = stc_phase_matrices[i].size();
    if (i == 0) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    int pre_rotation = 0;
    int additional_pt_rot = 0;
    if (i == 0) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = (1 << num_stages);
    } else if (i == num_stc_levels_ - 1) {
      pre_rotation = (1 << stc_stages_cumul);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << stc_stages_cumul);
      additional_pt_rot = (1 << (num_stages + stc_stages_cumul));
    }

    int pt_level = scalars_.stc_start_level - i;
    double pt_scale = encoder_.GetRescalePrimeProd(pt_level);
    stc_stages_cumul += num_stages;

    auto hoist_map = ConstructPlainHoistMap(stc_phase_matrices[i], bs, gs,
                                            pre_rotation, additional_pt_rot);

    bool bs_gs_swapped =
        (hoist_map.size() == 1 && hoist_map.begin()->first == 0);

    LinearTransformPhaseSpec spec;
    spec.pt_level = pt_level;
    spec.pt_scale = pt_scale;
    spec.bs = bs;
    spec.gs = gs;
    spec.bs_gs_swapped = bs_gs_swapped;
    spec.pre_rotation = pre_rotation;
    spec.additional_pt_rot = additional_pt_rot;
    spec.num_limbs = (pt_level + 1) + params_.alpha;

    if (bs_gs_swapped) {
      spec.bs_indices = {0};
      for (const auto& [bs_idx, _] : hoist_map.begin()->second) {
        spec.gs_indices.push_back(bs_idx);
      }
      std::sort(spec.gs_indices.begin(), spec.gs_indices.end());
      spec.pt_slab_word_offset.resize(spec.gs_indices.size(),
                                      std::vector<int64_t>(1, -1));
      for (size_t g = 0; g < spec.gs_indices.size(); ++g) {
        spec.pt_slab_word_offset[g][0] = current_slab_words;
        current_slab_words +=
            static_cast<int64_t>(spec.num_limbs) * params_.degree;
      }
    } else {
      std::set<int> bs_set;
      for (const auto& [gs_idx, bs_map] : hoist_map) {
        spec.gs_indices.push_back(gs_idx);
        for (const auto& [bs_idx, _] : bs_map) {
          bs_set.insert(bs_idx);
        }
      }
      std::sort(spec.gs_indices.begin(), spec.gs_indices.end());
      spec.bs_indices.assign(bs_set.begin(), bs_set.end());
      spec.pt_slab_word_offset.resize(
          spec.gs_indices.size(),
          std::vector<int64_t>(spec.bs_indices.size(), -1));
      for (size_t g = 0; g < spec.gs_indices.size(); ++g) {
        int gs_idx = spec.gs_indices[g];
        for (size_t b = 0; b < spec.bs_indices.size(); ++b) {
          int bs_idx = spec.bs_indices[b];
          if (hoist_map.count(gs_idx) && hoist_map.at(gs_idx).count(bs_idx)) {
            spec.pt_slab_word_offset[g][b] = current_slab_words;
            current_slab_words +=
                static_cast<int64_t>(spec.num_limbs) * params_.degree;
          }
        }
      }
    }
    plan.stc.push_back(std::move(spec));
  }

  plan.total_slab_words = current_slab_words;
  return plan;
}

template <unsigned W>
BootEvkPlan BootConsts<W>::GetBootEvkPlan(int num_slots) const {
  BootEvkPlan plan;
  int64_t current_byte_offset = 0;
  int max_num_slots = static_cast<int>(params_.degree / 2);
  bool full_slot = (num_slots == max_num_slots);

  auto add_evk = [&](BootEvkKind kind, int rot_index, int max_level, int num_q,
                     int num_aux, int beta) {
    int64_t words =
        2LL * beta * (num_q + num_aux) * static_cast<int64_t>(params_.degree);
    int64_t bytes = words * sizeof(Word);
    BootEvkSpec spec;
    spec.kind = kind;
    spec.rot_index = rot_index;
    spec.max_level = max_level;
    spec.num_q = num_q;
    spec.num_aux = num_aux;
    spec.beta = beta;
    spec.slab_byte_offset = current_byte_offset;
    spec.total_bytes = bytes;
    plan.evks.push_back(spec);
    current_byte_offset += bytes;
  };

  // 1. Special EVKs: dts, std, conj, relin
  // DtS key: level = -1, num_q = 1, num_aux = 1, beta = 1
  add_evk(BootEvkKind::kDts, 0, -1, 1, 1, 1);

  // StD key: level = max_level (25), num_q = 26, num_aux = 7, beta = 1
  // (pre-summed digit)
  add_evk(BootEvkKind::kStd, 0, scalars_.max_level, scalars_.max_level + 1,
          params_.alpha, 1);

  // Conjugation key: level = eval_mod_start_level (21), num_q = 22, num_aux =
  // 7, beta = 4
  int conj_num_q = scalars_.eval_mod_start_level + 1;
  int conj_beta = DivCeil(conj_num_q, params_.alpha);
  add_evk(BootEvkKind::kConj, 0, scalars_.eval_mod_start_level, conj_num_q,
          params_.alpha, conj_beta);

  // Relinearization key: level = max_level (25), num_q = 26, num_aux = 7, beta
  // = 4
  int relin_num_q = scalars_.max_level + 1;
  int relin_beta = DivCeil(relin_num_q, params_.alpha);
  add_evk(BootEvkKind::kRelin, 0, scalars_.max_level, relin_num_q,
          params_.alpha, relin_beta);

  // 2. Rotation keys
  std::map<int, int> req_rotations;

  // Trace rotations: for ns = num_slots; ns < max_num_slots; ns *= 2
  for (int ns = num_slots; ns < max_num_slots; ns *= 2) {
    req_rotations[ns] = scalars_.max_level;
  }

  // Linear transform phase rotations
  LinearTransformPlan lt_plan = GetLinearTransformPlan(num_slots);
  for (const auto& phase : lt_plan.cts) {
    for (int bs : phase.bs_indices) {
      if (bs != 0) {
        req_rotations[bs] = std::max(req_rotations[bs], phase.pt_level);
      }
    }
    for (int gs : phase.gs_indices) {
      if (gs != 0) {
        req_rotations[gs] = std::max(req_rotations[gs], phase.pt_level);
      }
    }
  }

  for (const auto& phase : lt_plan.stc) {
    for (int bs : phase.bs_indices) {
      if (bs != 0) {
        req_rotations[bs] = std::max(req_rotations[bs], phase.pt_level);
      }
    }
    for (int gs : phase.gs_indices) {
      if (gs != 0) {
        req_rotations[gs] = std::max(req_rotations[gs], phase.pt_level);
      }
    }
  }

  if (!full_slot) {
    req_rotations[num_slots] =
        std::max(req_rotations[num_slots], scalars_.end_level);
  }

  for (const auto& [rot_idx, max_level] : req_rotations) {
    int num_q = max_level + 1;
    int num_aux = params_.alpha;
    int beta = DivCeil(num_q, num_aux);
    add_evk(BootEvkKind::kRot, rot_idx, max_level, num_q, num_aux, beta);
  }

  plan.total_slab_bytes = current_byte_offset;
  return plan;
}

template <unsigned W>
std::vector<typename BootConsts<W>::Word> BootConsts<W>::PackBootConstsSlab(
    int num_slots) const {
  LinearTransformPlan plan = GetLinearTransformPlan(num_slots);
  std::vector<Word> slab(plan.total_slab_words, 0);

  auto pack_phases =
      [&](const std::vector<StripedMat>& phase_matrices,
          const std::vector<LinearTransformPhaseSpec>& phase_specs) {
        for (size_t p = 0; p < phase_specs.size(); ++p) {
          const auto& spec = phase_specs[p];
          auto hoist_map =
              ConstructPlainHoistMap(phase_matrices[p], spec.bs, spec.gs,
                                     spec.pre_rotation, spec.additional_pt_rot);

          if (spec.bs_gs_swapped) {
            for (size_t g = 0; g < spec.gs_indices.size(); ++g) {
              int bs_idx = spec.gs_indices[g];
              const auto& orig_msg = hoist_map.begin()->second.at(bs_idx);
              int msg_size = orig_msg.size();
              std::vector<Complex> rotated_msg(msg_size);
              for (int i = 0; i < msg_size; ++i) {
                rotated_msg[(i + bs_idx) % msg_size] = orig_msg[i];
              }
              auto ptxt = EncodeDiagonal(rotated_msg, spec.pt_level,
                                         spec.pt_scale, params_.alpha);
              int64_t offset = spec.pt_slab_word_offset[g][0];
              std::copy(ptxt.begin(), ptxt.end(), slab.data() + offset);
            }
          } else {
            for (size_t g = 0; g < spec.gs_indices.size(); ++g) {
              int gs_idx = spec.gs_indices[g];
              if (!hoist_map.count(gs_idx)) continue;
              for (size_t b = 0; b < spec.bs_indices.size(); ++b) {
                int bs_idx = spec.bs_indices[b];
                if (!hoist_map.at(gs_idx).count(bs_idx)) continue;
                const auto& msg = hoist_map.at(gs_idx).at(bs_idx);
                auto ptxt = EncodeDiagonal(msg, spec.pt_level, spec.pt_scale,
                                           params_.alpha);
                int64_t offset = spec.pt_slab_word_offset[g][b];
                std::copy(ptxt.begin(), ptxt.end(), slab.data() + offset);
              }
            }
          }
        }
      };

  auto cts_matrices = GenerateCtSPhaseMatrices(num_slots);
  pack_phases(cts_matrices, plan.cts);

  auto stc_matrices = GenerateStCPhaseMatrices(num_slots);
  pack_phases(stc_matrices, plan.stc);

  return slab;
}

template class RefBasisMap<32>;
template class RefBasisMap<64>;
template class RefEvalPolyNode<32>;
template class RefEvalPolyNode<64>;
template class BootConsts<32>;
template class BootConsts<64>;

}  // namespace mlir::heir::cheddar_ref
