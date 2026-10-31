#ifndef LIB_UTILS_CHEDDARREF_MONTARITH_H_
#define LIB_UTILS_CHEDDARREF_MONTARITH_H_

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// Traits defining integral word types and wide word types for bit-width W.
/// Cheddar uses 32-bit (uint32_t/int32_t/uint64_t/int64_t) and 64-bit
/// (uint64_t/int64_t/uint128_t/int128_t) arithmetic.
/// References:
/// - third_party/cheddar/include/core/Type.h:9-11
/// - third_party/cheddar/include/common/DoubleWord.h:8-55
template <unsigned W>
struct MontArithTraits;

template <>
struct MontArithTraits<32> {
  using Word = uint32_t;
  using SignedWord = int32_t;
  using DoubleWord = uint64_t;
  using SignedDoubleWord = int64_t;
};

template <>
struct MontArithTraits<64> {
  using Word = uint64_t;
  using SignedWord = int64_t;
  using DoubleWord = unsigned __int128;
  using SignedDoubleWord = __int128;
};

/// Header-only reference model of Cheddar's signed Montgomery modular
/// arithmetic, transliterating third_party/cheddar/include/common/Basic.cuh
/// and third_party/cheddar/include/common/PrimeUtils.h.
///
/// Modulus constraints (third_party/cheddar/include/core/Parameter.h:49-52):
/// - W = 32: odd prime modulus q < 2^31
/// - W = 64: odd prime modulus q < 2^63
template <unsigned W>
struct MontArith {
  static_assert(W == 32 || W == 64, "MontArith only supports W=32 or W=64");

  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;
  using DoubleWord = typename Traits::DoubleWord;
  using SignedDoubleWord = typename Traits::SignedDoubleWord;

  /// Wide signed multiplication: a * b.
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:36-41 (32-bit)
  /// and 52-59 (64-bit) (__mult_wide<int32_t> and __mult_wide<int64_t>).
  static constexpr SignedDoubleWord MultWide(SignedWord a, SignedWord b) {
    return static_cast<SignedDoubleWord>(a) * static_cast<SignedDoubleWord>(b);
  }

  /// Wide unsigned multiplication: a * b.
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:28-33 (32-bit)
  /// and 44-50 (64-bit) (__mult_wide<uint32_t> and __mult_wide<uint64_t>).
  static constexpr DoubleWord MultWide(Word a, Word b) {
    return static_cast<DoubleWord>(a) * static_cast<DoubleWord>(b);
  }

  /// Signed high-multiplication: returns the upper W bits of signed (a * b).
  /// Transliterates CUDA PTX / intrinsics __mulhi (32-bit) and __mul64hi
  /// (64-bit) used in third_party/cheddar/include/common/Basic.cuh:56, 86, 97,
  /// 132, 135.
  static constexpr SignedWord MulHi(SignedWord a, SignedWord b) {
    SignedDoubleWord prod = MultWide(a, b);
    return static_cast<SignedWord>(prod >> W);
  }

  /// Unsigned high-multiplication: returns the upper W bits of unsigned (a *
  /// b). Transliterates CUDA intrinsic __umul64hi used in
  /// third_party/cheddar/include/common/Basic.cuh:47.
  static constexpr Word MulHi(Word a, Word b) {
    DoubleWord prod = MultWide(a, b);
    return static_cast<Word>(prod >> W);
  }

  /// Lazy Montgomery modular reduction: computes (a * R^-1) mod q, returning
  /// a signed result in [-(q-1), q-1], i.e., strictly in (-q, q).
  ///
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:80-99
  /// (__montgomery_reduction_lazy).
  /// Formula: hi - mulhi(lo * q_inv, q).
  ///
  /// Range contract:
  /// - Input a must be in range [-q * 2^(W-1), q * 2^(W-1) - 1].
  /// - Modulus q must be an odd prime < 2^(W-1).
  /// - Output is guaranteed to be in range [-(q-1), q-1].
  static SignedWord MontReduceLazy(SignedDoubleWord a, Word q,
                                   SignedWord q_inv) {
    assert(q > 0 && (q & 1) == 1 &&
           "Modulus q must be an odd positive integer");
    assert(q < (Word(1) << (W - 1)) && "Modulus q must be < 2^(W-1)");

    SignedDoubleWord half_base = static_cast<SignedDoubleWord>(1) << (W - 1);
    SignedDoubleWord min_a = -static_cast<SignedDoubleWord>(q) * half_base;
    SignedDoubleWord max_a = static_cast<SignedDoubleWord>(q) * half_base - 1;
    assert(a >= min_a && a <= max_a &&
           "MontReduceLazy input a must be in range [-q*2^(W-1), q*2^(W-1)-1]");

    SignedWord lo = static_cast<SignedWord>(a);
    SignedWord hi = static_cast<SignedWord>(a >> W);

    // Mirrors third_party/cheddar/include/common/Basic.cuh:88-98:
    //   SignedWord temp = lo * q_inv;
    //   temp = mul_hi(temp, q);
    //   SignedWord res = hi - temp;
    // On CUDA, signed integer multiplication and subtraction wrap around modulo
    // 2^W in two's complement. In standard C++, signed overflow is undefined
    // behavior (UB). We mirror Basic.cuh's wrap semantics without UB by
    // computing in unsigned Word and casting back to SignedWord.
    SignedWord temp = static_cast<SignedWord>(static_cast<Word>(lo) *
                                              static_cast<Word>(q_inv));
    temp = MulHi(temp, static_cast<SignedWord>(q));
    SignedWord res = static_cast<SignedWord>(static_cast<Word>(hi) -
                                             static_cast<Word>(temp));

    assert(res > -static_cast<SignedWord>(q) &&
           res < static_cast<SignedWord>(q) &&
           "MontReduceLazy output must be in range [-(q-1), q-1]");
    return res;
  }

  /// Montgomery reduction: computes (a * R^-1) mod q and canonicalizes to [0,
  /// q). Transliterates third_party/cheddar/include/common/Basic.cuh:150-157
  /// (ReduceMontgomery).
  static Word MontReduce(SignedDoubleWord a, Word q, SignedWord q_inv) {
    SignedWord res = MontReduceLazy(a, q, q_inv);
    if (res < 0) res += static_cast<SignedWord>(q);
    assert(res >= 0 && static_cast<Word>(res) < q);
    return static_cast<Word>(res);
  }

  /// Lazy Montgomery multiplication: computes (a * b * R^-1) mod q for signed
  /// inputs, returning a signed result in [-(q-1), q-1].
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:113-137
  /// (__mult_montgomery_lazy).
  static SignedWord MultMontLazy(SignedWord a, SignedWord b, Word q,
                                 SignedWord q_inv) {
    SignedDoubleWord mult = MultWide(a, b);
    return MontReduceLazy(mult, q, q_inv);
  }

  /// Montgomery multiplication: computes (a * b * R^-1) mod q in [0, q).
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:251-260
  /// (MultMontgomery).
  static Word MultMont(Word a, Word b, Word q, SignedWord q_inv) {
    assert(a < q && "Input a must be in range [0, q-1]");
    assert(b < q && "Input b must be in range [0, q-1]");
    SignedWord res = MultMontLazy(static_cast<SignedWord>(a),
                                  static_cast<SignedWord>(b), q, q_inv);
    if (res < 0) res += static_cast<SignedWord>(q);
    assert(res >= 0 && static_cast<Word>(res) < q);
    return static_cast<Word>(res);
  }

  /// Modular addition: calculates (a + b) % q in [0, q).
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:168-174 (Add).
  static Word Add(Word a, Word b, Word q) {
    assert(a < q && "Input a must be in range [0, q-1]");
    assert(b < q && "Input b must be in range [0, q-1]");
    Word res = a + b;
    if (res >= q) res -= q;
    assert(res < q);
    return res;
  }

  /// Modular subtraction: calculates (a - b) % q in [0, q).
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:185-192 (Sub).
  static Word Sub(Word a, Word b, Word q) {
    assert(a < q && "Input a must be in range [0, q-1]");
    assert(b < q && "Input b must be in range [0, q-1]");
    SignedWord res = static_cast<SignedWord>(a) - static_cast<SignedWord>(b);
    if (res < 0) res += static_cast<SignedWord>(q);
    assert(res >= 0 && static_cast<Word>(res) < q);
    return static_cast<Word>(res);
  }

  /// Modular negation: calculates (-a) % q in [0, q).
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:217-222
  /// (Negate).
  static Word Negate(Word a, Word q) {
    assert(a < q && "Input a must be in range [0, q-1]");
    Word res = 0;
    if (a > 0) res = q - a;
    assert(res < q);
    return res;
  }

  /// Centers a in [0, q-1] into the signed range [-(q-1)/2, (q-1)/2].
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:232-239
  /// (Normalize).
  static SignedWord Normalize(Word a, Word q) {
    assert(a < q && "Input a must be in range [0, q-1]");
    SignedWord res = static_cast<SignedWord>(a);
    if (a > (q >> 1)) res -= static_cast<SignedWord>(q);
    assert(res >= -static_cast<SignedWord>(q >> 1) &&
           res <= static_cast<SignedWord>(q >> 1));
    return res;
  }

  /// Converts standard representation to Montgomery form: (a * 2^W) % q.
  /// Transliterates third_party/cheddar/include/common/PrimeUtils.h:242-248
  /// (ToMontgomery).
  static Word ToMont(Word a, Word q) {
    assert(a < q && "Input a must be in range [0, q-1]");
    return static_cast<Word>((static_cast<DoubleWord>(a) << W) % q);
  }

  /// Converts standard representation to Montgomery form using precomputed R^2
  /// mod q: MultMont(a, R^2, q, q_inv) == (a * R^2 * R^-1) % q == (a * R) % q.
  static Word ToMont(Word a, Word r2, Word q, SignedWord q_inv) {
    return MultMont(a, r2, q, q_inv);
  }

  /// Converts Montgomery representation back to standard form:
  /// (a_mont * R^-1) mod q.
  static Word FromMont(Word a_mont, Word q, SignedWord q_inv) {
    assert(a_mont < q && "Input a_mont must be in range [0, q-1]");
    return MontReduce(static_cast<SignedDoubleWord>(a_mont), q, q_inv);
  }

  /// Computes R mod q = 2^W mod q.
  static Word ComputeR(Word q) {
    assert(q > 0 && "Modulus q must be positive");
    return static_cast<Word>((static_cast<DoubleWord>(1) << W) % q);
  }

  /// Computes R^2 mod q = (2^W)^2 mod q.
  static Word ComputeR2(Word q) {
    Word r = ComputeR(q);
    return static_cast<Word>((static_cast<DoubleWord>(r) * r) % q);
  }

  /// Computes q^(-1) mod 2^W matching Cheddar's InvModBase sign convention.
  ///
  /// Transliterates third_party/cheddar/include/common/PrimeUtils.h:206-232
  /// (InvModBase) and 26-44 (__gcd_extended).
  ///
  /// Convention: q_inv satisfies (static_cast<Word>(q_inv) * q) == 1 mod 2^W.
  /// It is the POSITIVE multiplicative inverse q^-1 mod 2^W (NOT -q^-1),
  /// because Cheddar's lazy reduction formula subtracts mulhi: `hi -
  /// mulhi(...)`.
  static SignedWord ComputeQInv(Word q) {
    assert(q > 0 && (q & 1) == 1 &&
           "Modulus q must be an odd positive integer");
    assert(q < (Word(1) << (W - 1)) && "Modulus q must be < 2^(W-1)");

    Word base = ~Word(0);  // 2^W - 1
    Word quotient = base / q;
    Word remainder = (base % q) + 1;

    SignedWord x1 = 0;
    SignedWord y1 = 0;
    ExtendedGcd(static_cast<SignedWord>(remainder), static_cast<SignedWord>(q),
                x1, y1);

    SignedWord inv_q = static_cast<SignedWord>(static_cast<Word>(y1) -
                                               static_cast<Word>(quotient) *
                                                   static_cast<Word>(x1));
    assert(static_cast<Word>(inv_q) * q == Word(1) &&
           "q_inv * q must equal 1 mod 2^W");
    return inv_q;
  }

  /// Performs bit reversal of index x over the specified number of bits.
  /// Transliterates third_party/cheddar/include/common/Basic.cuh:269-272
  /// (BitReverse).
  /// Parameter `bits` must be in range [0, W].
  static Word BitReverse(Word x, unsigned int bits) {
    assert(bits <= W && "bits must be in range [0, W]");
    if (bits == 0) return 0;
    if constexpr (W == 32) {
      return __builtin_bitreverse32(x) >> (32 - bits);
    } else {
      return __builtin_bitreverse64(x) >> (64 - bits);
    }
  }

 private:
  /// Extended Euclidean algorithm to find x, y satisfying ax + by = gcd(a, b).
  /// Transliterates third_party/cheddar/include/common/PrimeUtils.h:26-44
  /// (__gcd_extended).
  static SignedWord ExtendedGcd(SignedWord a, SignedWord b, SignedWord& x,
                                SignedWord& y) {
    if (a == 0) {
      x = 0;
      y = 1;
      return b;
    }
    SignedWord x1 = 0;
    SignedWord y1 = 0;
    SignedWord gcd = ExtendedGcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return gcd;
  }
};

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_MONTARITH_H_
