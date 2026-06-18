#ifndef LIB_UTILS_POLYNOMIAL_NTT_H_
#define LIB_UTILS_POLYNOMIAL_NTT_H_

#include <cstdint>
#include <vector>

namespace mlir {
namespace heir {
namespace polynomial {

/// Reverse the bits of an integer 'n' up to 'bitWidth' bits.
uint32_t reverseBits(uint32_t n, uint32_t bitWidth);

/// Perform the forward RNS-NTT (Number Theoretic Transform) in-place on
/// 'coeffs'. 'coeffs' represents a polynomial in Z_q[x]/(x^n + 1), where n is a
/// power of 2. 'modulus' is the prime modulus q, which must be < 2^64.
/// 'rootOfUnity' is a primitive 2n-th root of unity modulo q. The resulting
/// coefficients are stored in bit-reversed evaluation order.
void nttInPlace(std::vector<uint64_t>& coeffs, uint64_t modulus,
                uint64_t rootOfUnity);

/// Perform the inverse RNS-NTT in-place on 'coeffs'.
/// 'coeffs' is in bit-reversed evaluation order.
/// 'modulus' is the prime modulus q, which must be < 2^64.
/// 'rootOfUnity' is a primitive 2n-th root of unity modulo q.
/// The resulting coefficients represent the polynomial in standard order.
void inttInPlace(std::vector<uint64_t>& coeffs, uint64_t modulus,
                 uint64_t rootOfUnity);

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_NTT_H_
