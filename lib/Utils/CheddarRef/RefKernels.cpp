#include "lib/Utils/CheddarRef/RefKernels.h"

#include <cassert>
#include <cstddef>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

template <unsigned W>
void RefKernels<W>::NTT(Word* buf, const Tables<W>& tables,
                        LimbRange limb_range, bool montgomery_conversion) {
  size_t start = limb_range.start;
  size_t end = (limb_range.end > 0) ? limb_range.end : tables.twiddle_limbs;
  assert(start < end && end <= tables.twiddle_limbs);

  size_t degree = tables.degree;
  std::vector<SignedWord> local(degree);

  for (size_t l = start; l < end; ++l) {
    Word* poly = buf + (l - start) * degree;
    Word q = tables.primes[l];
    SignedWord q_inv = tables.q_inv[l];
    Word r2 = tables.R2_mod_q[l];
    const Word* w_table = tables.psi_rev_mont.data() + l * degree;

    // Optional Montgomery conversion (NTT.cu:642-648)
    if (montgomery_conversion) {
      for (size_t i = 0; i < degree; ++i) {
        assert(poly[i] < q && "Input must be canonical residue in [0, q)");
        local[i] =
            MontArith<W>::MultMontLazy(static_cast<SignedWord>(poly[i]),
                                       static_cast<SignedWord>(r2), q, q_inv);
      }
    } else {
      for (size_t i = 0; i < degree; ++i) {
        assert(poly[i] < q && "Input must be canonical residue in [0, q)");
        local[i] = static_cast<SignedWord>(poly[i]);
      }
    }

    // Radix-2 Cooley-Tukey Decimation-in-Time (CT-DIT)
    size_t t = degree / 2;
    size_t m = 1;
    while (t >= 1) {
      for (size_t i = 0; i < m; ++i) {
        Word w = w_table[m + i];
        for (size_t j = 0; j < t; ++j) {
          size_t idx0 = 2 * i * t + j;
          size_t idx1 = idx0 + t;
          ButterflyNTT(local[idx0], local[idx1], w, q, q_inv);
        }
      }
      t /= 2;
      m *= 2;
    }

    // Final canonicalization to [0, q) (NTT.cu:445-450)
    for (size_t i = 0; i < degree; ++i) {
      if (local[i] < 0) local[i] += static_cast<SignedWord>(q);
      assert(local[i] >= 0 && static_cast<Word>(local[i]) < q);
      poly[i] = static_cast<Word>(local[i]);
    }
  }
}

template <unsigned W>
void RefKernels<W>::INTT(Word* buf, const Tables<W>& tables,
                         LimbRange limb_range, bool inv_default_true,
                         INTTPostProcess post_process,
                         const Word* const_operand) {
  size_t start = limb_range.start;
  size_t end = (limb_range.end > 0) ? limb_range.end : tables.twiddle_limbs;
  assert(start < end && end <= tables.twiddle_limbs);

  size_t degree = tables.degree;
  std::vector<SignedWord> local(degree);

  for (size_t l = start; l < end; ++l) {
    Word* poly = buf + (l - start) * degree;
    Word q = tables.primes[l];
    SignedWord q_inv = tables.q_inv[l];
    const Word* inv_w_table = tables.psi_inv_rev_mont.data() + l * degree;

    for (size_t i = 0; i < degree; ++i) {
      assert(poly[i] < q && "Input must be canonical residue in [0, q)");
      local[i] = static_cast<SignedWord>(poly[i]);
    }

    // Radix-2 Gentleman-Sande Decimation-in-Frequency (GS-DIF)
    size_t t = 1;
    size_t m = degree / 2;
    while (m >= 1) {
      for (size_t i = 0; i < m; ++i) {
        Word w = inv_w_table[m + i];
        for (size_t j = 0; j < t; ++j) {
          size_t idx0 = 2 * i * t + j;
          size_t idx1 = idx0 + t;
          ButterflyINTT(local[idx0], local[idx1], w, q, q_inv);
        }
      }
      t *= 2;
      m /= 2;
    }

    // Epilogue / Post-processing
    if (post_process == INTTPostProcess::None) {
      Word c = inv_default_true ? tables.N_inv[l] : tables.N_inv_mont[l];
      for (size_t i = 0; i < degree; ++i) {
        poly[i] = MultConst(local[i], c, q, q_inv);
      }
    } else if (post_process == INTTPostProcess::MultConst) {
      assert(const_operand != nullptr && "const_operand must not be null");
      Word c = const_operand[l - start];
      for (size_t i = 0; i < degree; ++i) {
        poly[i] = MultConst(local[i], c, q, q_inv);
      }
    } else if (post_process == INTTPostProcess::MultConstNormalize) {
      assert(const_operand != nullptr && "const_operand must not be null");
      Word c = const_operand[l - start];
      for (size_t i = 0; i < degree; ++i) {
        poly[i] = MultConstNormalize(local[i], c, q, q_inv);
      }
    }
  }
}

template class RefKernels<32>;
template class RefKernels<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
