#include <cstdint>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/Dialect/CKKS/Transforms/runner/relinearize_test_data.h"
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_relin(StridedMemRefType<int64_t, 3>* result,
                             StridedMemRefType<int64_t, 3>* ct,
                             StridedMemRefType<int64_t, 4>* ksk);
void _mlir_ciface_key_to_ntt(StridedMemRefType<int64_t, 2>* result,
                             StridedMemRefType<int64_t, 2>* poly);
}

namespace {

constexpr int64_t kNumQP = kNumQ + 1;

// The relinearization key in the [num_digits][2][N][num_limbs] layout, with
// each polynomial converted to NTT form as the lowered @test_relin expects.
std::vector<int64_t> relinKeyInNttForm() {
  const int64_t polySize = kN * kNumQP;
  const int64_t numPolys = kNumDigits * 2;
  std::vector<int64_t> key(kRelinKey, kRelinKey + numPolys * polySize);
  for (int64_t p = 0; p < numPolys; ++p) {
    std::vector<int64_t> poly(key.begin() + p * polySize,
                              key.begin() + (p + 1) * polySize);
    StridedMemRefType<int64_t, 2> input(poly.data(), poly.data(), 0,
                                        {kN, kNumQP}, {kNumQP, 1});
    StridedMemRefType<int64_t, 2> output;
    _mlir_ciface_key_to_ntt(&output, &input);
    for (int64_t i = 0; i < kN; ++i) {
      for (int64_t l = 0; l < kNumQP; ++l) {
        key[p * polySize + i * kNumQP + l] =
            output.data[output.offset + i * output.strides[0] +
                        l * output.strides[1]];
      }
    }
    free(output.basePtr);
  }
  return key;
}

TEST(RelinearizeTest, PreservesPhaseWithinNoiseBound) {
  std::vector<int64_t> ct(kCiphertext, kCiphertext + 3 * kN * kNumQ);
  std::vector<int64_t> ksk = relinKeyInNttForm();
  StridedMemRefType<int64_t, 3> ctRef(ct.data(), ct.data(), 0, {3, kN, kNumQ},
                                      {kN * kNumQ, kNumQ, 1});
  StridedMemRefType<int64_t, 4> kskRef(
      ksk.data(), ksk.data(), 0, {kNumDigits, 2, kN, kNumQP},
      {2 * kN * kNumQP, kN * kNumQP, kNumQP, 1});
  StridedMemRefType<int64_t, 3> result;
  _mlir_ciface_test_relin(&result, &ctRef, &kskRef);

  ASSERT_EQ(result.sizes[0], 2);
  ASSERT_EQ(result.sizes[1], kN);
  ASSERT_EQ(result.sizes[2], kNumQ);
  auto output = [&](int64_t component, int64_t i, int64_t limb) -> uint64_t {
    return result.data[result.offset + component * result.strides[0] +
                       i * result.strides[1] + limb * result.strides[2]];
  };

  for (int64_t l = 0; l < kNumQ; ++l) {
    const uint64_t q = kQ[l];

    // Compute the phase d0 + d1 * s in Z_q[x] / (x^N + 1). The secret key has
    // coefficients in {-1, 0, 1}, so this only needs modular additions.
    std::vector<uint64_t> phase(kN);
    for (int64_t i = 0; i < kN; ++i) {
      phase[i] = output(0, i, l);
    }
    for (int64_t j = 0; j < kN; ++j) {
      if (kSecretKey[j] == 0) continue;
      for (int64_t i = 0; i < kN; ++i) {
        const uint64_t d1 = output(1, i, l);
        const int64_t k = (i + j) % kN;
        const bool negate = (kSecretKey[j] < 0) != (i + j >= kN);
        phase[k] = (phase[k] + (negate ? q - d1 : d1)) % q;
      }
    }

    for (int64_t i = 0; i < kN; ++i) {
      const uint64_t expected = kExpectedPhase[i * kNumQ + l];
      const uint64_t diff = (phase[i] + q - expected) % q;
      const uint64_t noise = diff > q / 2 ? q - diff : diff;
      ASSERT_LE(noise, static_cast<uint64_t>(kNoiseBound))
          << "limb " << l << ", coefficient " << i;
    }
  }

  free(result.basePtr);
}

}  // namespace
