#include "lib/Utils/CheddarRef/Encoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/RefKernels.h"
#include "lib/Utils/CheddarRef/Tables.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir::heir::cheddar_ref {
namespace {

int Log2Ceil(int a) {
  assert(a > 0);
  int ret = 0;
  while ((1 << ret) < a) {
    ++ret;
  }
  return ret;
}

unsigned BitReverseInt(unsigned a, int num_bits) {
  static const unsigned char table[] = {
      0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0,
      0x30, 0xb0, 0x70, 0xf0, 0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
      0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 0x04, 0x84, 0x44, 0xc4,
      0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
      0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc,
      0x3c, 0xbc, 0x7c, 0xfc, 0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
      0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 0x0a, 0x8a, 0x4a, 0xca,
      0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
      0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6,
      0x36, 0xb6, 0x76, 0xf6, 0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
      0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe, 0x01, 0x81, 0x41, 0xc1,
      0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
      0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9,
      0x39, 0xb9, 0x79, 0xf9, 0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
      0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 0x0d, 0x8d, 0x4d, 0xcd,
      0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
      0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3,
      0x33, 0xb3, 0x73, 0xf3, 0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
      0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 0x07, 0x87, 0x47, 0xc7,
      0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
      0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf,
      0x3f, 0xbf, 0x7f, 0xff,
  };
  unsigned ret = 0;
  for (size_t i = 0; i < sizeof(unsigned); ++i) {
    ret = (ret << 8) | static_cast<unsigned>(table[(a >> (i * 8)) & 0xff]);
  }
  return ret >> (sizeof(unsigned) * 8 - num_bits);
}

template <typename T>
void BitReverseVector(std::vector<T>& data) {
  int num_slots = data.size();
  int log_slots = Log2Ceil(num_slots);
  data.resize(1 << log_slots);
  for (int i = 0; i < num_slots; ++i) {
    int j = BitReverseInt(i, log_slots);
    if (i < j) {
      std::swap(data[i], data[j]);
    }
  }
}

constexpr unsigned kBigIntBits = 4096;

/// Big integer struct backed by llvm::APInt to replace GMP mpz_t without
/// external dependency.
struct Mpz {
  llvm::APInt val;
  Mpz() : val(kBigIntBits, 0) {}
  explicit Mpz(uint64_t v) : val(kBigIntBits, v) {}
  explicit Mpz(double v) : val(kBigIntBits, 0) { SetDouble(v); }

  void SetUint64(uint64_t v) { val = llvm::APInt(kBigIntBits, v); }

  void SetDouble(double d) {
    if (std::isnan(d) || std::isinf(d) || d == 0.0) {
      val = llvm::APInt(kBigIntBits, 0);
      return;
    }
    bool neg = (d < 0);
    double abs_d = std::abs(d);
    if (abs_d < 1.0) {
      val = llvm::APInt(kBigIntBits, 0);
      return;
    }
    int exp = 0;
    double frac = std::frexp(abs_d, &exp);
    // frac is in [0.5, 1.0)
    uint64_t mantissa = static_cast<uint64_t>(std::ldexp(frac, 53));
    int shift = exp - 53;
    llvm::APInt ap(kBigIntBits, mantissa);
    if (shift > 0) {
      ap = ap.shl(shift);
    } else if (shift < 0) {
      ap = ap.lshr(-shift);
    }
    if (neg) {
      ap = -ap;
    }
    val = ap;
  }
};

inline void mpz_set_ui(llvm::APInt& rop, uint64_t op) {
  rop = llvm::APInt(kBigIntBits, op);
}

inline void mpz_set_d(llvm::APInt& rop, double d) {
  Mpz m(d);
  rop = m.val;
}

inline void mpz_mul(llvm::APInt& rop, const llvm::APInt& op1,
                    const llvm::APInt& op2) {
  rop = op1 * op2;
}

inline void mpz_add(llvm::APInt& rop, const llvm::APInt& op1,
                    const llvm::APInt& op2) {
  rop = op1 + op2;
}

inline void mpz_sub(llvm::APInt& rop, const llvm::APInt& op1,
                    const llvm::APInt& op2) {
  rop = op1 - op2;
}

inline void mpz_mod(llvm::APInt& rop, const llvm::APInt& n,
                    const llvm::APInt& d) {
  if (n.isNegative()) {
    llvm::APInt rem = (-n).urem(d);
    rop = rem.isZero() ? rem : (d - rem);
  } else {
    rop = n.urem(d);
  }
}

inline int mpz_cmp(const llvm::APInt& op1, const llvm::APInt& op2) {
  if (op1.slt(op2)) return -1;
  if (op1.sgt(op2)) return 1;
  return 0;
}

inline void mpz_fdiv_q_2exp(llvm::APInt& q, const llvm::APInt& n, unsigned b) {
  q = n.ashr(b);
}

inline uint64_t mpz_fdiv_ui(const llvm::APInt& n, uint64_t d) {
  llvm::APInt div(kBigIntBits, d);
  llvm::APInt rem(kBigIntBits, 0);
  if (n.isNegative()) {
    rem = (-n).urem(div);
    if (!rem.isZero()) {
      rem = div - rem;
    }
  } else {
    rem = n.urem(div);
  }
  return rem.getZExtValue();
}

inline double mpz_get_d(const llvm::APInt& op) {
  if (op.isZero()) return 0.0;
  bool neg = op.isNegative();
  llvm::APInt abs_op = neg ? -op : op;
  unsigned active_bits = abs_op.getActiveBits();
  if (active_bits <= 53) {
    double d = static_cast<double>(abs_op.getZExtValue());
    return neg ? -d : d;
  }
  unsigned shift = active_bits - 53;
  llvm::APInt top53 = abs_op.lshr(shift);
  double mantissa = static_cast<double>(top53.getZExtValue());
  double d = std::ldexp(mantissa, shift);
  return neg ? -d : d;
}

}  // namespace

template <unsigned W>
Encoder<W>::Encoder(const Params<W>& params, const Tables<W>& tables)
    : params_(params),
      tables_(tables),
      degree_(params.degree),
      M_(static_cast<int>(params.degree * 2)) {
  InitTwiddles();
  InitGaloisFactors();
  InitScaleChain();
}

template <unsigned W>
void Encoder<W>::InitTwiddles() {
  twiddle_factors_.resize(M_);
  for (int i = 0; i < M_; ++i) {
    twiddle_factors_[i] = std::polar(1.0, 2.0 * M_PI * i / M_);
  }
}

template <unsigned W>
void Encoder<W>::InitGaloisFactors() {
  int max_num_slots = degree_ / 2;
  galois_factors_.resize(max_num_slots + 1);
  galois_factors_[max_num_slots] = 1;
  galois_factors_[0] = 1;
  for (int i = 1; i < max_num_slots; ++i) {
    galois_factors_[i] = (galois_factors_[i - 1] * 5) % M_;
  }
}

template <unsigned W>
void Encoder<W>::InitScaleChain() {
  int max_level = params_.scale_info.max_level;
  int default_encryption_level = params_.scale_info.default_encryption_level;
  int log_default_scale = params_.scale_info.log_default_scale;

  rescale_prime_prod_.resize(max_level + 1);
  scale_.resize(default_encryption_level + 1);

  double base_scale =
      static_cast<double>(static_cast<uint64_t>(1) << log_default_scale);
  scale_[0] = base_scale;
  double new_scale = base_scale;

  for (int i = 1; i <= max_level; ++i) {
    NPInfo np_upper = LevelToNP(i);
    NPInfo np_lower = LevelToNP(i - 1);
    int ter_diff = np_upper.num_ter - np_lower.num_ter;
    int main_diff = np_upper.num_main - np_lower.num_main;

    double scale_numerator = 1.0;
    double scale_denominator = 1.0;

    if (ter_diff >= 0) {
      for (int j = 0; j < ter_diff; ++j) {
        scale_numerator *= params_.ter_primes[np_lower.num_ter + j];
      }
    } else {
      for (int j = 0; j < -ter_diff; ++j) {
        scale_denominator *= params_.ter_primes[np_upper.num_ter + j];
      }
    }

    if (main_diff >= 0) {
      for (int j = 0; j < main_diff; ++j) {
        scale_numerator *= params_.main_primes[np_lower.num_main + j];
      }
    } else {
      for (int j = 0; j < -main_diff; ++j) {
        scale_denominator *= params_.main_primes[np_upper.num_main + j];
      }
    }

    double prod = scale_numerator / scale_denominator;
    rescale_prime_prod_[i] = prod;

    if (i <= default_encryption_level) {
      new_scale = std::sqrt(new_scale * prod);
      scale_[i] = new_scale;
    }
  }
}

template <unsigned W>
double Encoder<W>::GetScale(int level) const {
  if (level == -1) level = 0;
  assert(level >= 0 && level <= params_.scale_info.default_encryption_level &&
         "GetScale: Invalid level");
  return scale_.at(level);
}

template <unsigned W>
double Encoder<W>::GetRescalePrimeProd(int level) const {
  assert(level > 0 && level <= params_.scale_info.max_level &&
         "GetRescalePrimeProd: Invalid level");
  return rescale_prime_prod_.at(level);
}

template <unsigned W>
NPInfo Encoder<W>::LevelToNP(int level, int num_aux) const {
  assert(num_aux >= 0 && num_aux <= params_.alpha && "Invalid num_aux");
  if (level == -1) {
    int num_main =
        params_.level_config.at(0).first - params_.additional_base.first;
    int num_ter =
        params_.level_config.at(0).second - params_.additional_base.second;
    return NPInfo{num_main, num_ter, num_aux};
  }
  assert(level >= 0 && level < static_cast<int>(params_.level_config.size()) &&
         "Invalid level");
  const auto& config = params_.level_config.at(level);
  return NPInfo{config.first, config.second, num_aux};
}

template <unsigned W>
typename Encoder<W>::Complex Encoder<W>::GetTwiddleFactor(int index) const {
  assert(index >= 0 && index < M_ && "Invalid twiddle factor index");
  return twiddle_factors_[index];
}

template <unsigned W>
int Encoder<W>::GetGaloisFactor(int i) const {
  assert(i >= 0 && i <= static_cast<int>(degree_ / 2) &&
         "Invalid Galois factor index");
  return galois_factors_.at(i);
}

template <unsigned W>
void Encoder<W>::SpecialIFFT(std::vector<Complex>& data) const {
  int num_slots = data.size();
  assert(num_slots == (1 << Log2Ceil(num_slots)) &&
         "Power of 2 num slots only");

  for (int stride = num_slots / 2; stride >= 1; stride /= 2) {
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M_ / st8;
    for (int i = 0; i < num_slots; i += stride_group_size) {
      for (int j = 0; j < stride; ++j) {
        int twiddle_index = (st8 - (GetGaloisFactor(j) % st8)) * gap;
        auto x = data[i + j] + data[i + j + stride];
        auto y = data[i + j] - data[i + j + stride];
        y *= twiddle_factors_[twiddle_index];
        data[i + j] = x;
        data[i + j + stride] = y;
      }
    }
  }
  BitReverseVector(data);
  for (int i = 0; i < num_slots; ++i) {
    data[i] /= num_slots;
  }
}

template <unsigned W>
void Encoder<W>::SpecialFFT(std::vector<Complex>& data) const {
  int num_slots = data.size();
  assert(num_slots == (1 << Log2Ceil(num_slots)) &&
         "Power of 2 num slots only");
  BitReverseVector(data);

  for (int stride = 1; stride < num_slots; stride *= 2) {
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M_ / st8;
    for (int i = 0; i < num_slots; i += stride_group_size) {
      for (int j = 0; j < stride; ++j) {
        int twiddle_index = (GetGaloisFactor(j) % st8) * gap;
        auto x = data[i + j];
        auto y = data[i + j + stride];
        y *= twiddle_factors_[twiddle_index];
        data[i + j] = x + y;
        data[i + j + stride] = x - y;
      }
    }
  }
}

template <unsigned W>
void Encoder<W>::EncodeStandardCoeffs(Word* out_coeffs, int level, double scale,
                                      const std::vector<Complex>& message,
                                      int num_aux) const {
  int msg_length = message.size();
  int num_slots = 1 << Log2Ceil(msg_length);
  std::vector<Complex> padded_msg(num_slots);
  std::copy(message.begin(), message.end(), padded_msg.begin());

  SpecialIFFT(padded_msg);

  int degree = degree_;
  int half_degree = degree / 2;
  int gap = half_degree / num_slots;

  assert(num_slots <= half_degree && "Too many slots");

  NPInfo np = LevelToNP(level, num_aux);
  int num_q = np.num_main + np.num_ter;
  int num_total_primes = np.GetNumTotal();
  int max_num_ter = params_.ter_primes.size();
  size_t q_start = max_num_ter - np.num_ter;
  size_t aux_start = max_num_ter + params_.main_primes.size();

  std::fill(out_coeffs, out_coeffs + num_total_primes * degree, 0);

  Mpz big_real, big_imag;

  for (int i = 0; i < num_slots; ++i) {
    Complex value = padded_msg[i] * scale;
    double real = value.real();
    double imag = value.imag();

    mpz_set_d(big_real.val, real);
    mpz_set_d(big_imag.val, imag);

    int real_degree_index = i * gap;
    int imag_degree_index = real_degree_index + half_degree;

    for (int j = 0; j < num_q; ++j) {
      Word prime = params_.all_primes[q_start + j];
      Word real_mod = static_cast<Word>(mpz_fdiv_ui(big_real.val, prime));
      Word imag_mod = static_cast<Word>(mpz_fdiv_ui(big_imag.val, prime));

      out_coeffs[real_degree_index + j * degree] = real_mod;
      out_coeffs[imag_degree_index + j * degree] = imag_mod;
    }
    for (int k = 0; k < np.num_aux; ++k) {
      Word prime = params_.all_primes[aux_start + k];
      Word real_mod = static_cast<Word>(mpz_fdiv_ui(big_real.val, prime));
      Word imag_mod = static_cast<Word>(mpz_fdiv_ui(big_imag.val, prime));

      out_coeffs[real_degree_index + (num_q + k) * degree] = real_mod;
      out_coeffs[imag_degree_index + (num_q + k) * degree] = imag_mod;
    }
  }
}

template <unsigned W>
std::vector<typename Encoder<W>::Word> Encoder<W>::EncodeStandardCoeffs(
    int level, double scale, const std::vector<Complex>& message,
    int num_aux) const {
  NPInfo np = LevelToNP(level, num_aux);
  std::vector<Word> coeffs(np.GetNumTotal() * degree_);
  EncodeStandardCoeffs(coeffs.data(), level, scale, message, num_aux);
  return coeffs;
}

template <unsigned W>
void Encoder<W>::Encode(Word* out_ntt, int level, double scale,
                        const std::vector<Complex>& message,
                        int num_aux) const {
  NPInfo np = LevelToNP(level, num_aux);
  int num_q = np.num_main + np.num_ter;
  int max_num_ter = params_.ter_primes.size();
  size_t q_start = max_num_ter - np.num_ter;
  size_t aux_start = max_num_ter + params_.main_primes.size();

  EncodeStandardCoeffs(out_ntt, level, scale, message, num_aux);

  if (num_q > 0) {
    RefKernels<W>::NTT(out_ntt, tables_, LimbRange{q_start, q_start + num_q},
                       /*montgomery_conversion=*/true);
  }
  if (np.num_aux > 0) {
    RefKernels<W>::NTT(out_ntt + num_q * degree_, tables_,
                       LimbRange{aux_start, aux_start + np.num_aux},
                       /*montgomery_conversion=*/true);
  }
}

template <unsigned W>
std::vector<typename Encoder<W>::Word> Encoder<W>::Encode(
    int level, double scale, const std::vector<Complex>& message,
    int num_aux) const {
  NPInfo np = LevelToNP(level, num_aux);
  std::vector<Word> ntt_poly(np.GetNumTotal() * degree_);
  Encode(ntt_poly.data(), level, scale, message, num_aux);
  return ntt_poly;
}

template <unsigned W>
void Encoder<W>::EncodeConstant(Word* out_const, int level, double scale,
                                double number, int num_aux) const {
  Mpz big_int(number * scale);

  NPInfo np = LevelToNP(level, num_aux);
  int num_q = np.num_main + np.num_ter;
  int max_num_ter = params_.ter_primes.size();
  size_t q_start = max_num_ter - np.num_ter;
  size_t aux_start = max_num_ter + params_.main_primes.size();

  for (int i = 0; i < num_q; ++i) {
    size_t idx = q_start + i;
    Word prime = params_.all_primes[idx];
    Word mod_val = static_cast<Word>(mpz_fdiv_ui(big_int.val, prime));
    Word r_mod_q = tables_.R_mod_q[idx];
    out_const[i] = Tables<W>::MultMod(mod_val, r_mod_q, prime);
  }
  for (int k = 0; k < np.num_aux; ++k) {
    size_t idx = aux_start + k;
    Word prime = params_.all_primes[idx];
    Word mod_val = static_cast<Word>(mpz_fdiv_ui(big_int.val, prime));
    Word r_mod_q = tables_.R_mod_q[idx];
    out_const[num_q + k] = Tables<W>::MultMod(mod_val, r_mod_q, prime);
  }
}

template <unsigned W>
std::vector<typename Encoder<W>::Word> Encoder<W>::EncodeConstant(
    int level, double scale, double number, int num_aux) const {
  NPInfo np = LevelToNP(level, num_aux);
  std::vector<Word> cx(np.GetNumTotal());
  EncodeConstant(cx.data(), level, scale, number, num_aux);
  return cx;
}

template <unsigned W>
std::vector<typename Encoder<W>::Complex> Encoder<W>::Decode(
    const Word* in_ntt, int level, double scale, int num_slots) const {
  NPInfo np = LevelToNP(level, /*num_aux=*/0);
  int num_total_primes = np.GetNumTotal();
  int max_num_ter = params_.ter_primes.size();
  size_t start_limb = max_num_ter - np.num_ter;
  size_t end_limb = start_limb + num_total_primes;

  // 1. Create iCRT constants: icrt1[i] = ((prod_{j != i} q_j) * N)^-1 mod q_i
  std::vector<Word> icrt1(num_total_primes);
  for (int i = 0; i < num_total_primes; ++i) {
    Word mod_prime = params_.all_primes[start_limb + i];
    Word prod = 1;
    for (int j = 0; j < num_total_primes; ++j) {
      if (i == j) continue;
      Word other_prime = params_.all_primes[start_limb + j];
      prod = Tables<W>::MultMod(prod, other_prime % mod_prime, mod_prime);
    }
    prod = Tables<W>::MultMod(prod, static_cast<Word>(degree_ % mod_prime),
                              mod_prime);
    icrt1[i] = Tables<W>::InvMod(prod, mod_prime);
  }

  // 2. In-place INTT + MultConst(icrt1)
  std::vector<Word> intt_res(in_ntt, in_ntt + num_total_primes * degree_);
  RefKernels<W>::INTTAndMultConst(intt_res.data(), tables_, icrt1.data(),
                                  LimbRange{start_limb, end_limb});

  // 3. Reconstruct complex vector using GMP CRT
  int half_degree = degree_ / 2;
  int gap = half_degree / num_slots;

  std::vector<Mpz> big_primes(num_total_primes);
  Mpz prime_prod(static_cast<uint64_t>(1));

  for (int i = 0; i < num_total_primes; ++i) {
    mpz_set_ui(big_primes[i].val, params_.all_primes[start_limb + i]);
    mpz_mul(prime_prod.val, prime_prod.val, big_primes[i].val);
  }

  Mpz half_prime_prod;
  mpz_fdiv_q_2exp(half_prime_prod.val, prime_prod.val, 1);

  std::vector<Mpz> prods(num_total_primes);
  for (int j = 0; j < num_total_primes; ++j) {
    mpz_set_ui(prods[j].val, 1);
    for (int k = 0; k < num_total_primes; ++k) {
      if (j == k) continue;
      mpz_mul(prods[j].val, prods[j].val, big_primes[k].val);
    }
  }

  std::vector<Complex> data(num_slots);
  Mpz tmp, real_icrt, imag_icrt, val_mpz;

  for (int i = 0; i < num_slots; ++i) {
    mpz_set_ui(real_icrt.val, 0);
    mpz_set_ui(imag_icrt.val, 0);

    for (int j = 0; j < num_total_primes; ++j) {
      uint64_t r_val = static_cast<uint64_t>(intt_res[j * degree_ + i * gap]);
      uint64_t i_val =
          static_cast<uint64_t>(intt_res[j * degree_ + i * gap + half_degree]);

      mpz_set_ui(val_mpz.val, r_val);
      mpz_mul(tmp.val, val_mpz.val, prods[j].val);
      mpz_add(real_icrt.val, real_icrt.val, tmp.val);

      mpz_set_ui(val_mpz.val, i_val);
      mpz_mul(tmp.val, val_mpz.val, prods[j].val);
      mpz_add(imag_icrt.val, imag_icrt.val, tmp.val);
    }

    mpz_mod(real_icrt.val, real_icrt.val, prime_prod.val);
    if (mpz_cmp(real_icrt.val, half_prime_prod.val) > 0) {
      mpz_sub(real_icrt.val, real_icrt.val, prime_prod.val);
    }

    mpz_mod(imag_icrt.val, imag_icrt.val, prime_prod.val);
    if (mpz_cmp(imag_icrt.val, half_prime_prod.val) > 0) {
      mpz_sub(imag_icrt.val, imag_icrt.val, prime_prod.val);
    }

    double real = mpz_get_d(real_icrt.val) / scale;
    double imag = mpz_get_d(imag_icrt.val) / scale;
    data[i] = Complex(real, imag);
  }

  SpecialFFT(data);
  return data;
}

template <unsigned W>
std::vector<typename Encoder<W>::Complex> Encoder<W>::DecodeStandardCoeffs(
    const Word* in_coeffs, int level, double scale, int num_slots) const {
  NPInfo np = LevelToNP(level, /*num_aux=*/0);
  int num_total_primes = np.GetNumTotal();
  int max_num_ter = params_.ter_primes.size();
  size_t start_limb = max_num_ter - np.num_ter;

  // Multiply by (prod_{j != i} q_j)^-1 mod q_i
  std::vector<Word> scaled_coeffs(num_total_primes * degree_);
  for (int i = 0; i < num_total_primes; ++i) {
    Word mod_prime = params_.all_primes[start_limb + i];
    Word prod = 1;
    for (int j = 0; j < num_total_primes; ++j) {
      if (i == j) continue;
      Word other_prime = params_.all_primes[start_limb + j];
      prod = Tables<W>::MultMod(prod, other_prime % mod_prime, mod_prime);
    }
    Word inv_prod = Tables<W>::InvMod(prod, mod_prime);
    for (size_t c = 0; c < degree_; ++c) {
      scaled_coeffs[i * degree_ + c] =
          Tables<W>::MultMod(in_coeffs[i * degree_ + c], inv_prod, mod_prime);
    }
  }

  int half_degree = degree_ / 2;
  int gap = half_degree / num_slots;

  std::vector<Mpz> big_primes(num_total_primes);
  Mpz prime_prod(static_cast<uint64_t>(1));

  for (int i = 0; i < num_total_primes; ++i) {
    mpz_set_ui(big_primes[i].val, params_.all_primes[start_limb + i]);
    mpz_mul(prime_prod.val, prime_prod.val, big_primes[i].val);
  }

  Mpz half_prime_prod;
  mpz_fdiv_q_2exp(half_prime_prod.val, prime_prod.val, 1);

  std::vector<Mpz> prods(num_total_primes);
  for (int j = 0; j < num_total_primes; ++j) {
    mpz_set_ui(prods[j].val, 1);
    for (int k = 0; k < num_total_primes; ++k) {
      if (j == k) continue;
      mpz_mul(prods[j].val, prods[j].val, big_primes[k].val);
    }
  }

  std::vector<Complex> data(num_slots);
  Mpz tmp, real_icrt, imag_icrt, val_mpz;

  for (int i = 0; i < num_slots; ++i) {
    mpz_set_ui(real_icrt.val, 0);
    mpz_set_ui(imag_icrt.val, 0);

    for (int j = 0; j < num_total_primes; ++j) {
      uint64_t r_val =
          static_cast<uint64_t>(scaled_coeffs[j * degree_ + i * gap]);
      uint64_t i_val = static_cast<uint64_t>(
          scaled_coeffs[j * degree_ + i * gap + half_degree]);

      mpz_set_ui(val_mpz.val, r_val);
      mpz_mul(tmp.val, val_mpz.val, prods[j].val);
      mpz_add(real_icrt.val, real_icrt.val, tmp.val);

      mpz_set_ui(val_mpz.val, i_val);
      mpz_mul(tmp.val, val_mpz.val, prods[j].val);
      mpz_add(imag_icrt.val, imag_icrt.val, tmp.val);
    }

    mpz_mod(real_icrt.val, real_icrt.val, prime_prod.val);
    if (mpz_cmp(real_icrt.val, half_prime_prod.val) > 0) {
      mpz_sub(real_icrt.val, real_icrt.val, prime_prod.val);
    }

    mpz_mod(imag_icrt.val, imag_icrt.val, prime_prod.val);
    if (mpz_cmp(imag_icrt.val, half_prime_prod.val) > 0) {
      mpz_sub(imag_icrt.val, imag_icrt.val, prime_prod.val);
    }

    double real = mpz_get_d(real_icrt.val) / scale;
    double imag = mpz_get_d(imag_icrt.val) / scale;
    data[i] = Complex(real, imag);
  }

  SpecialFFT(data);
  return data;
}

template class Encoder<32>;
template class Encoder<64>;

}  // namespace mlir::heir::cheddar_ref
