#ifndef LIB_UTILS_CHEDDARREF_ENCODER_H_
#define LIB_UTILS_CHEDDARREF_ENCODER_H_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/RefKernels.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir::heir::cheddar_ref {

/// Information about prime partition at a given level.
struct NPInfo {
  int num_main = 0;
  int num_ter = 0;
  int num_aux = 0;

  int GetNumQ() const { return num_main + num_ter; }
  int GetNumTotal() const { return num_main + num_ter + num_aux; }
};

/// Reference implementation of Cheddar's Encoder / Decoder on CPU.
/// Operates on floating-point complex messages and converts them to
/// standard polynomial coefficients or Montgomery NTT evaluation domain
/// using arbitrary-precision arithmetic (GMP mpz_t).
template <unsigned W>
class Encoder {
 public:
  static_assert(W == 32 || W == 64, "Encoder only supports W=32 or W=64");

  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;
  using Complex = std::complex<double>;

  /// Constructs an Encoder from CKKS parameters and tables.
  Encoder(const Params<W>& params, const Tables<W>& tables);

  /// Canonical CKKS encode scale at the given level.
  double GetScale(int level) const;

  /// Rescale prime product factor at the given level.
  double GetRescalePrimeProd(int level) const;

  /// Returns prime partition for given level and num_aux.
  NPInfo LevelToNP(int level, int num_aux = 0) const;

  /// Returns twiddle factor for index i in [0, 2*degree).
  Complex GetTwiddleFactor(int index) const;

  /// Returns read-only reference to all 2*degree twiddle factors.
  const std::vector<Complex>& GetTwiddleFactors() const {
    return twiddle_factors_;
  }

  /// Returns Galois factor (5^i mod 2*degree).
  int GetGaloisFactor(int i) const;

  /// In-place special IFFT on complex message vector (size must be power of 2
  /// <= degree / 2).
  void SpecialIFFT(std::vector<Complex>& data) const;

  /// In-place special FFT on complex message vector (size must be power of 2 <=
  /// degree / 2).
  void SpecialFFT(std::vector<Complex>& data) const;

  /// Encodes complex message to standard-form polynomial coefficients (before
  /// NTT). Output buffer `out_coeffs` has shape [num_primes, degree] flattened
  /// row-major.
  void EncodeStandardCoeffs(Word* out_coeffs, int level, double scale,
                            const std::vector<Complex>& message,
                            int num_aux = 0) const;

  std::vector<Word> EncodeStandardCoeffs(int level, double scale,
                                         const std::vector<Complex>& message,
                                         int num_aux = 0) const;

  /// Encodes complex message directly to NTT evaluation domain (Montgomery
  /// form). Output buffer `out_ntt` has shape [num_primes, degree] flattened
  /// row-major.
  void Encode(Word* out_ntt, int level, double scale,
              const std::vector<Complex>& message, int num_aux = 0) const;

  std::vector<Word> Encode(int level, double scale,
                           const std::vector<Complex>& message,
                           int num_aux = 0) const;

  /// Encodes a real scalar constant into RNS residues in Montgomery form.
  /// Output buffer `out_const` has size num_primes.
  void EncodeConstant(Word* out_const, int level, double scale, double number,
                      int num_aux = 0) const;

  std::vector<Word> EncodeConstant(int level, double scale, double number,
                                   int num_aux = 0) const;

  /// Decodes a plaintext in NTT evaluation domain back into complex message
  /// slots. `in_ntt` has shape [num_primes, degree]. `num_slots` must be power
  /// of 2 <= degree / 2.
  std::vector<Complex> Decode(const Word* in_ntt, int level, double scale,
                              int num_slots) const;

  /// Decodes standard-form polynomial coefficients back into complex message
  /// slots.
  std::vector<Complex> DecodeStandardCoeffs(const Word* in_coeffs, int level,
                                            double scale, int num_slots) const;

 private:
  const Params<W>& params_;
  const Tables<W>& tables_;
  size_t degree_;
  int M_;

  std::vector<Complex> twiddle_factors_;
  std::vector<int> galois_factors_;
  std::vector<double> rescale_prime_prod_;
  std::vector<double> scale_;

  void InitTwiddles();
  void InitGaloisFactors();
  void InitScaleChain();
};

using Encoder32 = Encoder<32>;
using Encoder64 = Encoder<64>;

}  // namespace mlir::heir::cheddar_ref

#endif  // LIB_UTILS_CHEDDARREF_ENCODER_H_
