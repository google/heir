#ifndef LIB_UTILS_CHEDDARREF_REFKERNELS_H_
#define LIB_UTILS_CHEDDARREF_REFKERNELS_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// ============================================================================
/// Cheddar NTT / INTT Mathematical and Representation Contract
/// ============================================================================
///
/// 1. Forward Transform (NTT):
///    - Radix-2 Cooley-Tukey Decimation-in-Time (CT-DIT) butterfly network.
///    - Butterfly formula
///    (third_party/cheddar/include/core/NTTUtils.cuh:19-35):
///        mult = MontReduceLazy(b * w, q, q_inv)  // in (-q, q)
///        if (mult < 0) mult += q                 // in [0, q)
///        if (a < 0) a += q                       // in [0, q)
///        b_new = a - mult                        // in (-q, q)
///        a_new = (a - q) + mult                  // in [-q, q)
///    - Twiddle factors:
///        Uses psi_rev table in Montgomery form (psi_rev_mont[m + i]),
///        where psi = g^((q-1)/2N) with smallest generator g.
///        psi_rev[j] is bit-reversed, and psi_rev_mont[j] = (psi_rev[j] * R)
///        mod q.
///    - Input / Output ordering and domain:
///        Input: Standard natural order.
///        Output: Bit-reversed order in Montgomery evaluation domain.
///    - Montgomery conversion semantics (NTT.cu:642-648):
///        When montgomery_conversion == true:
///          Input elements are premultiplied by R^2 mod q via MontReduceLazy:
///          (x * R^2) * R^-1 = x * R mod q, converting them to Montgomery form.
///        When montgomery_conversion == false:
///          Input elements are assumed to already be in Montgomery form.
///    - Canonicalization:
///        Lazy range [-q, q) maintained internally across all butterfly stages;
///        at the very end of all NTT stages, any negative value is
///        canonicalized: if (x < 0) x += q, producing residues in [0, q).
///
/// 2. Inverse Transform (INTT):
///    - Radix-2 Gentleman-Sande Decimation-in-Frequency (GS-DIF) butterfly
///    network.
///    - Butterfly formula
///    (third_party/cheddar/include/core/NTTUtils.cuh:46-60):
///        if (a < 0) a += q                       // in [0, q)
///        if (b < 0) b += q                       // in [0, q)
///        diff = a - b                            // in (-q, q)
///        a_new = (a - q) + b                     // in [-q, q)
///        b_new = MontReduceLazy(diff * w, q, q_inv) // in (-q, q)
///    - Twiddle factors:
///        Uses psi_inv_rev_mont table (psi_inv_rev_mont[m + i]),
///        with psi_inv = psi^-1 mod q in bit-reversed Montgomery form.
///    - Input / Output ordering and domain:
///        Input: Bit-reversed order in evaluation domain.
///        Output: Standard natural order in coefficient domain.
///    - Montgomery conversion semantics (NTT.cu:720-724):
///        Default (inv_default_true == true, i.e. montgomery_conversion == true
///        in Cheddar):
///          Multiplies the butterfly result by plain N^-1 mod q via MultConst:
///          (x * R * N) * N^-1 * R^-1 = x mod q,
///          leaving the result in standard (non-Montgomery) canonical
///          representation in [0, q).
///        When inv_default_true == false (i.e. montgomery_conversion == false
///        in Cheddar):
///          Multiplies by (N^-1 * R) mod q via MultConst:
///          (x * R * N) * (N^-1 * R) * R^-1 = x * R mod q,
///          leaving the result in Montgomery form in [0, q).
///
/// 3. Epilogue Variants (NTT.cu:788-812, NTTUtils.cuh:363-391):
///    - MultConst:
///        Multiplies by a per-limb constant c via Montgomery reduction:
///        temp = MontReduceLazy(local * c, q, q_inv);
///        if (temp < 0) temp += q;
///        Outputs canonical residue in [0, q).
///    - MultConstNormalize:
///        Multiplies by a per-limb constant c via Montgomery reduction,
///        then centers the canonical residue:
///        temp = MontReduceLazy(local * c, q, q_inv);
///        if (temp < 0) temp += q;
///        if (temp > (q >> 1)) temp -= q;
///        Outputs centered signed value in [-(q-1)/2, (q-1)/2] (stored as two's
///        complement).
///
/// 4. Limb-skip ranges (NTTForModUp, NTT.cu:815-860):
///    - Operates on a subset of limbs, skipping prime indices in [skip_start,
///    skip_end).
///    - In RefKernels, limb selection is handled via LimbRange {start, end}.
/// ============================================================================

/// Post-processing / epilogue mode for INTT.
enum class INTTPostProcess {
  None,
  MultConst,
  MultConstNormalize,
};

/// Contiguous range of limbs [start, end) to transform.
struct LimbRange {
  size_t start = 0;
  size_t end = 0;
};

template <unsigned W>
class RefKernels {
 public:
  static_assert(W == 32 || W == 64, "RefKernels only supports W=32 or W=64");

  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  /// Epilogue helper: computes MultConst on a single value:
  /// MontReduce(a * c) canonicalized to [0, q).
  static Word MultConst(SignedWord a, Word c, Word q, SignedWord q_inv) {
    SignedWord temp =
        MontArith<W>::MultMontLazy(a, static_cast<SignedWord>(c), q, q_inv);
    if (temp < 0) temp += static_cast<SignedWord>(q);
    assert(temp >= 0 && static_cast<Word>(temp) < q);
    return static_cast<Word>(temp);
  }

  /// Epilogue helper: computes MultConstNormalize on a single value:
  /// MontReduce(a * c) canonicalized to [0, q), then centered to
  /// [-(q-1)/2, (q-1)/2].
  /// Returned as Word via two's-complement bit-pattern.
  static Word MultConstNormalize(SignedWord a, Word c, Word q,
                                 SignedWord q_inv) {
    SignedWord temp =
        MontArith<W>::MultMontLazy(a, static_cast<SignedWord>(c), q, q_inv);
    if (temp < 0) temp += static_cast<SignedWord>(q);
    assert(temp >= 0 && static_cast<Word>(temp) < q);
    if (static_cast<Word>(temp) > (q >> 1)) {
      temp -= static_cast<SignedWord>(q);
    }
    return static_cast<Word>(temp);
  }

  /// Reverses the bits of a polynomial of size `degree` in place.
  static void BitReverse(Word* poly, size_t degree, unsigned log_degree) {
    for (size_t i = 0; i < degree; ++i) {
      size_t j = MontArith<W>::BitReverse(i, log_degree);
      if (i < j) {
        std::swap(poly[i], poly[j]);
      }
    }
  }

  /// Forward NTT: transforms polynomial coefficients from standard order to
  /// bit-reversed order in Montgomery evaluation domain.
  /// Operates in-place on `buf` for limbs in `limb_range`.
  /// If `limb_range.end == 0`, defaults to `tables.twiddle_limbs`.
  static void NTT(Word* buf, const Tables<W>& tables, LimbRange limb_range = {},
                  bool montgomery_conversion = false);

  /// Inverse NTT: transforms bit-reversed evaluations back to standard order.
  /// Operates in-place on `buf` for limbs in `limb_range`.
  /// If `limb_range.end == 0`, defaults to `tables.twiddle_limbs`.
  static void INTT(Word* buf, const Tables<W>& tables,
                   LimbRange limb_range = {}, bool inv_default_true = true,
                   INTTPostProcess post_process = INTTPostProcess::None,
                   const Word* const_operand = nullptr);

  /// Convenience wrapper for INTT followed by per-limb constant multiplication.
  static void INTTAndMultConst(Word* buf, const Tables<W>& tables,
                               const Word* const_operand,
                               LimbRange limb_range = {},
                               bool normalize = false) {
    INTT(buf, tables, limb_range, /*inv_default_true=*/true,
         normalize ? INTTPostProcess::MultConstNormalize
                   : INTTPostProcess::MultConst,
         const_operand);
  }

  /// NTTForModUp: transforms limbs in `limb_range` skipping [skip_start,
  /// skip_end).
  static void NTTForModUp(Word* buf, const Tables<W>& tables, size_t skip_start,
                          size_t skip_end, LimbRange limb_range = {}) {
    size_t range_start = limb_range.start;
    size_t range_end =
        (limb_range.end > 0) ? limb_range.end : tables.twiddle_limbs;

    if (skip_start > range_start) {
      NTT(buf, tables, LimbRange{range_start, std::min(skip_start, range_end)},
          /*montgomery_conversion=*/false);
    }
    if (skip_end < range_end) {
      size_t second_start = std::max(skip_end, range_start);
      size_t buf_offset = (second_start - range_start) * tables.degree;
      NTT(buf + buf_offset, tables, LimbRange{second_start, range_end},
          /*montgomery_conversion=*/false);
    }
  }

 private:
  /// CT-DIT butterfly helper.
  static void ButterflyNTT(SignedWord& a, SignedWord& b, Word w, Word q,
                           SignedWord q_inv) {
    if (a < 0) a += static_cast<SignedWord>(q);
    assert(a >= 0 && a < static_cast<SignedWord>(q));
    SignedWord mult =
        MontArith<W>::MultMontLazy(b, static_cast<SignedWord>(w), q, q_inv);
    if (mult < 0) mult += static_cast<SignedWord>(q);
    assert(mult >= 0 && mult < static_cast<SignedWord>(q));
    b = a - mult;
    a = (a - static_cast<SignedWord>(q)) + mult;
    assert(b > -static_cast<SignedWord>(q) && b < static_cast<SignedWord>(q));
    assert(a >= -static_cast<SignedWord>(q) && a < static_cast<SignedWord>(q));
  }

  /// GS-DIF butterfly helper.
  static void ButterflyINTT(SignedWord& a, SignedWord& b, Word w, Word q,
                            SignedWord q_inv) {
    if (a < 0) a += static_cast<SignedWord>(q);
    if (b < 0) b += static_cast<SignedWord>(q);
    assert(a >= 0 && a < static_cast<SignedWord>(q));
    assert(b >= 0 && b < static_cast<SignedWord>(q));
    SignedWord diff = a - b;
    assert(diff > -static_cast<SignedWord>(q) &&
           diff < static_cast<SignedWord>(q));
    a = (a - static_cast<SignedWord>(q)) + b;
    assert(a >= -static_cast<SignedWord>(q) && a < static_cast<SignedWord>(q));
    b = MontArith<W>::MultMontLazy(diff, static_cast<SignedWord>(w), q, q_inv);
    assert(b > -static_cast<SignedWord>(q) && b < static_cast<SignedWord>(q));
  }
};

using RefKernels32 = RefKernels<32>;
using RefKernels64 = RefKernels<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_REFKERNELS_H_
