// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!digit_decomp = !openfhe.digit_decomp

// CHECK: CiphertextT test_fast_rot(CryptoContextT [[cc:.*]], CiphertextT [[ct:.*]]) {
// CHECK:  const auto& [[digit_decomp:.*]] = [[cc]]->EvalFastRotationPrecompute([[ct]]);
// CHECK:  const std::vector<size_t> [[v4:.*]] =
// CHECK:  std::vector<CiphertextT> [[v5:.*]](4);
// CHECK:  #pragma omp parallel for
// CHECK:  for (auto [[v7:.*]] = 0; [[v7]] < 4; ++[[v7]]) {
// CHECK:    size_t [[v9:.*]] = [[v4]][[[v7]]];
// CHECK:    const auto& [[ct1:.*]] = [[cc]]->EvalFastRotation([[ct]], [[v9]], 2 * [[cc]]->GetRingDimension(), [[digit_decomp]]);
// CHECK:    const std::vector<CiphertextT> [[v10:.*]] = {[[ct]]1};
// CHECK:    [[v5]][[[v7]]] = [[v10]][0];
// CHECK:  }
// CHECK:  const auto& [[ct2:.*]] = [[v5]][0];
module attributes {scheme.ckks} {
  func.func @test_fast_rot(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %digit_decomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %from_elements = tensor.from_elements %c1, %c2, %c3, %c4 : tensor<4xindex>
    %0 = tensor.empty() : tensor<4x!ct>
    %1 = scf.forall (%arg0) in (4) shared_outs(%arg1 = %0) -> (tensor<4x!ct>) {
      %extracted_9 = tensor.extract %from_elements[%arg0] : tensor<4xindex>
      %ct_10 = openfhe.fast_rotation %cc, %ct, %extracted_9, %digit_decomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !digit_decomp) -> !ct
      %from_elements_11 = tensor.from_elements %ct_10 : tensor<1x!ct>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %from_elements_11 into %arg1[%arg0] [1] [1] : tensor<1x!ct> into tensor<4x!ct>
      }
    }
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %1[%c0] : tensor<4x!ct>
    %c1_0 = arith.constant 1 : index
    %extracted_1 = tensor.extract %1[%c1_0] : tensor<4x!ct>
    %c2_2 = arith.constant 2 : index
    %extracted_3 = tensor.extract %1[%c2_2] : tensor<4x!ct>
    %c3_4 = arith.constant 3 : index
    %extracted_5 = tensor.extract %1[%c3_4] : tensor<4x!ct>
    %ct_6 = openfhe.add %cc, %extracted, %extracted_1 : (!cc, !ct, !ct) -> !ct
    %ct_7 = openfhe.add %cc, %ct_6, %extracted_3 : (!cc, !ct, !ct) -> !ct
    %ct_8 = openfhe.add %cc, %ct_7, %extracted_5 : (!cc, !ct, !ct) -> !ct
    return %ct_8 : !ct
  }
}
