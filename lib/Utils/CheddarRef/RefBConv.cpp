#include "lib/Utils/CheddarRef/RefBConv.h"

#include <cstddef>
#include <cstdint>

#include "lib/Utils/CheddarRef/MontArith.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

template <unsigned W>
void RefBConv<W>::BConv(Word* dst, const SignedWord* src,
                        const SignedWord* bconv_table, size_t src_len,
                        size_t dst_len, size_t degree, const Word* dst_primes,
                        const SignedWord* dst_inv_primes, size_t skip_start,
                        size_t skip_end, bool periodic_recenter) {
  constexpr size_t kMaxNumAccum = 4;

  for (size_t m = 0; m < dst_len; ++m) {
    size_t prime_index = m;
    if (skip_start < skip_end && prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    Word p = dst_primes[prime_index];
    SignedWord p_inv = dst_inv_primes[prime_index];

    // prime_th = p * 2^W
    SignedDoubleWord prime_th = static_cast<SignedDoubleWord>(p) << W;
    SignedDoubleWord prime_th_half = prime_th >> 1;

    const SignedWord* table_row = bconv_table + m * src_len;

    for (size_t x = 0; x < degree; ++x) {
      if (periodic_recenter) {
        SignedDoubleWord accum = 0;
        size_t num_accumulated = 0;

        for (size_t s = 0; s < src_len; ++s) {
          SignedWord s_val = src[s * degree + x];
          SignedWord t_val = table_row[s];
          accum += MontArith<W>::MultWide(s_val, t_val);
          num_accumulated++;

          if (num_accumulated >= kMaxNumAccum) {
            if (accum < 0) accum += prime_th;
            if (accum >= prime_th_half) accum -= prime_th;
            num_accumulated = 0;
          }
        }

        dst[prime_index * degree + x] =
            MontArith<W>::MontReduce(accum, p, p_inv);
      } else {
        if constexpr (W == 32) {
          // Accumulate in exact 128-bit integer without periodic re-centering
          __int128_t wide_accum = 0;
          for (size_t s = 0; s < src_len; ++s) {
            __int128_t s_val = src[s * degree + x];
            __int128_t t_val = table_row[s];
            wide_accum += s_val * t_val;
          }
          // Reduce mod (p * 2^32) to bring into [-p*2^31, p*2^31)
          __int128_t mod_val = static_cast<__int128_t>(p) << 32;
          __int128_t rem = wide_accum % mod_val;
          if (rem < 0) rem += mod_val;
          if (rem >= (mod_val >> 1)) rem -= mod_val;
          SignedDoubleWord accum = static_cast<SignedDoubleWord>(rem);
          dst[prime_index * degree + x] =
              MontArith<W>::MontReduce(accum, p, p_inv);
        } else {
          // For W=64, double word is 128-bit, re-center at the end
          SignedDoubleWord accum = 0;
          for (size_t s = 0; s < src_len; ++s) {
            SignedWord s_val = src[s * degree + x];
            SignedWord t_val = table_row[s];
            accum += MontArith<W>::MultWide(s_val, t_val);
            if ((s + 1) % kMaxNumAccum == 0 || s + 1 == src_len) {
              if (accum < 0) accum += prime_th;
              if (accum >= prime_th_half) accum -= prime_th;
            }
          }
          dst[prime_index * degree + x] =
              MontArith<W>::MontReduce(accum, p, p_inv);
        }
      }
    }
  }
}

template class RefBConv<32>;
template class RefBConv<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
