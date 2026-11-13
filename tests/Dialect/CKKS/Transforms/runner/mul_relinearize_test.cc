#include <cstdint>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/Dialect/CKKS/Transforms/runner/relinearize_test_data.h"
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_mul_relin(StridedMemRefType<int64_t, 3>* result,
                                 StridedMemRefType<int64_t, 3>* a,
                                 StridedMemRefType<int64_t, 3>* b,
                                 StridedMemRefType<int64_t, 4>* ksk);
void _mlir_ciface_key_to_ntt(StridedMemRefType<int64_t, 2>* result,
                             StridedMemRefType<int64_t, 2>* poly);
void _mlir_ciface_ct_to_ntt(StridedMemRefType<int64_t, 2>* result,
                            StridedMemRefType<int64_t, 2>* poly);
void _mlir_ciface_ct_from_ntt(StridedMemRefType<int64_t, 2>* result,
                              StridedMemRefType<int64_t, 2>* poly);
}

namespace {

constexpr int64_t kNumQP = kNumQ + 1;

using Converter = void (*)(StridedMemRefType<int64_t, 2>*,
                           StridedMemRefType<int64_t, 2>*);

// Applies `convert` to each [kN][numLimbs] polynomial in `data`, in place.
void convertEachPoly(std::vector<int64_t>& data, int64_t numLimbs,
                     Converter convert) {
  const int64_t polySize = kN * numLimbs;
  const int64_t numPolys = data.size() / polySize;
  for (int64_t p = 0; p < numPolys; ++p) {
    // Copy the polynomial out, since compiled code may write into its input.
    std::vector<int64_t> poly(data.begin() + p * polySize,
                              data.begin() + (p + 1) * polySize);
    StridedMemRefType<int64_t, 2> input(poly.data(), poly.data(), 0,
                                        {kN, numLimbs}, {numLimbs, 1});
    StridedMemRefType<int64_t, 2> output;
    convert(&output, &input);
    for (int64_t i = 0; i < kN; ++i) {
      for (int64_t l = 0; l < numLimbs; ++l) {
        data[p * polySize + i * numLimbs + l] =
            output.data[output.offset + i * output.strides[0] +
                        l * output.strides[1]];
      }
    }
    free(output.basePtr);
  }
}

TEST(MulRelinearizeTest, PreservesPhaseWithinNoiseBound) {
  // The lowered @test_mul_relin takes its inputs and returns its result in
  // NTT (eval) form, so convert the inputs here and the result below.
  const int64_t ctSize = 2 * kN * kNumQ;
  std::vector<int64_t> ksk(kRelinKey, kRelinKey + kNumDigits * 2 * kN * kNumQP);
  convertEachPoly(ksk, kNumQP, _mlir_ciface_key_to_ntt);
  std::vector<int64_t> a(kCiphertextA, kCiphertextA + ctSize);
  convertEachPoly(a, kNumQ, _mlir_ciface_ct_to_ntt);
  std::vector<int64_t> b(kCiphertextB, kCiphertextB + ctSize);
  convertEachPoly(b, kNumQ, _mlir_ciface_ct_to_ntt);

  StridedMemRefType<int64_t, 3> aRef(a.data(), a.data(), 0, {2, kN, kNumQ},
                                     {kN * kNumQ, kNumQ, 1});
  StridedMemRefType<int64_t, 3> bRef(b.data(), b.data(), 0, {2, kN, kNumQ},
                                     {kN * kNumQ, kNumQ, 1});
  StridedMemRefType<int64_t, 4> kskRef(
      ksk.data(), ksk.data(), 0, {kNumDigits, 2, kN, kNumQP},
      {2 * kN * kNumQP, kN * kNumQP, kNumQP, 1});
  StridedMemRefType<int64_t, 3> result;
  _mlir_ciface_test_mul_relin(&result, &aRef, &bRef, &kskRef);

  ASSERT_EQ(result.sizes[0], 2);
  ASSERT_EQ(result.sizes[1], kN);
  ASSERT_EQ(result.sizes[2], kNumQ);

  // Copy the result into a contiguous [2][kN][kNumQ] vector and convert it
  // back to coefficient form.
  std::vector<int64_t> out(ctSize);
  for (int64_t c = 0; c < 2; ++c) {
    for (int64_t i = 0; i < kN; ++i) {
      for (int64_t l = 0; l < kNumQ; ++l) {
        out[(c * kN + i) * kNumQ + l] =
            result.data[result.offset + c * result.strides[0] +
                        i * result.strides[1] + l * result.strides[2]];
      }
    }
  }
  free(result.basePtr);
  convertEachPoly(out, kNumQ, _mlir_ciface_ct_from_ntt);

  auto output = [&](int64_t component, int64_t i, int64_t limb) -> uint64_t {
    return out[(component * kN + i) * kNumQ + limb];
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
      const uint64_t expected = kExpectedMulPhase[i * kNumQ + l];
      const uint64_t diff = (phase[i] + q - expected) % q;
      const uint64_t noise = diff > q / 2 ? q - diff : diff;
      ASSERT_LE(noise, static_cast<uint64_t>(kNoiseBound))
          << "limb " << l << ", coefficient " << i;
    }
  }
}

}  // namespace
