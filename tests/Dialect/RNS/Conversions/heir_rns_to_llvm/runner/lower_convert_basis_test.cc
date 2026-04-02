#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

using ConvertBasisTestFn = void (*)(StridedMemRefType<int64_t, 1>*,
                                    StridedMemRefType<int64_t, 1>*);

extern "C" {
void _mlir_ciface_test_convert_basis_5_35(StridedMemRefType<int64_t, 1>* result,
                                          StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_5_57(StridedMemRefType<int64_t, 1>* result,
                                          StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_7_37(StridedMemRefType<int64_t, 1>* result,
                                          StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_7_57(StridedMemRefType<int64_t, 1>* result,
                                          StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_3_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_5_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_7_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_35_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_37_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_57_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_53_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_73_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
void _mlir_ciface_test_convert_basis_75_357(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<int64_t, 1>* arg0);
}

struct Case {
  std::vector<int64_t> src_basis;
  std::vector<int64_t> dst_basis;
  ConvertBasisTestFn fn;
};

std::vector<Case> cases = {
    {{5}, {3, 5}, _mlir_ciface_test_convert_basis_5_35},
    {{5}, {5, 7}, _mlir_ciface_test_convert_basis_5_57},
    {{7}, {3, 7}, _mlir_ciface_test_convert_basis_7_37},
    {{7}, {5, 7}, _mlir_ciface_test_convert_basis_7_57},
    {{3}, {3, 5, 7}, _mlir_ciface_test_convert_basis_3_357},
    {{5}, {3, 5, 7}, _mlir_ciface_test_convert_basis_5_357},
    {{7}, {3, 5, 7}, _mlir_ciface_test_convert_basis_7_357},
    {{3, 5}, {3, 5, 7}, _mlir_ciface_test_convert_basis_35_357},
    {{3, 7}, {3, 5, 7}, _mlir_ciface_test_convert_basis_37_357},
    {{5, 7}, {3, 5, 7}, _mlir_ciface_test_convert_basis_57_357},
    {{5, 3}, {3, 5, 7}, _mlir_ciface_test_convert_basis_53_357},
    {{7, 3}, {3, 5, 7}, _mlir_ciface_test_convert_basis_73_357},
    {{7, 5}, {3, 5, 7}, _mlir_ciface_test_convert_basis_75_357}};

TEST(LowerConvertBasisTest, TestConvertBasis) {
  StridedMemRefType<int64_t, 1> actualResult;
  for (const Case& c : cases) {
    // The product of the input basis moduli must fit into an int64_t
    int64_t inputModulus = 1;
    for (auto& q : c.src_basis) {
      inputModulus *= q;
    }
    for (int64_t x = 0; x < inputModulus; x++) {
      int64_t y = x;
      // we choose to lift to the centered representative
      // if (y > inputModulus / 2) {
      //   y -= inputModulus;
      // }
      std::vector<int64_t> expectedResult;
      for (auto& q : c.dst_basis) {
        expectedResult.push_back(y % q);
      }

      // compute result via MLIR
      std::vector<int64_t> residues;
      residues.reserve(c.src_basis.size());
      for (int64_t q : c.src_basis) {
        residues.push_back(y % q);
      }
      StridedMemRefType<int64_t, 1> memref = StridedMemRefType<int64_t, 1>(
          residues.data(), residues.data(), 0, residues.size(), 1);

      c.fn(&actualResult, &memref);
      std::vector<int64_t> actualValues(
          actualResult.data, actualResult.data + actualResult.sizes[0]);
      EXPECT_EQ(actualValues, expectedResult);
    }
  }
  free(actualResult.basePtr);
}
