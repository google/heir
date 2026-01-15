// RUN: heir-opt %s --boolean-vectorize | FileCheck %s

!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

// CHECK: func.func @test_fast_rot(%[[cc:.*]]: ![[cc:.*]], %[[input1:.*]]: ![[ct:.*]]) -> ![[ct]] {
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK:    %[[precomp:.*]] = openfhe.fast_rotation_precompute %[[cc]], %[[input1]]
// CHECK:    %[[from_elements:.*]] = tensor.from_elements %[[c1]], %[[c2]], %[[c3]], %[[c4]] : tensor<4xindex>
// CHECK:    %[[v0:.*]] = tensor.empty() : tensor<4x![[ct]]>
// CHECK:    %[[v1:.*]] = scf.forall (%[[arg0:.*]]) in (4) shared_outs(%[[arg1:.*]] = %[[v0]]) -> (tensor<4x![[ct]]>) {
// CHECK:      %[[extracted_9:.*]] = tensor.extract %[[from_elements]][%arg0] : tensor<4xindex>
// CHECK:      %[[ct_10:.*]] = openfhe.fast_rotation %[[cc]], %[[ct]], %[[extracted_9]], %[[precomp]]
// CHECK:      %[[from_elements_11:.*]] = tensor.from_elements %[[ct_10]] : tensor<1x![[ct]]>
// CHECK:      scf.forall.in_parallel {
// CHECK:        tensor.parallel_insert_slice %[[from_elements_11]] into %[[arg1]][%[[arg0]]] [1] [1] : tensor<1x![[ct]]> into tensor<4x![[ct]]>
// CHECK:      }
// CHECK:    }
module attributes {scheme.ckks} {
  func.func @test_fast_rot(%cc: !cc, %input1: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %input1 : (!cc, !ct) -> !openfhe.digit_decomp
    %res1 = openfhe.fast_rotation %cc, %input1, %c1, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !openfhe.digit_decomp) -> !ct
    %res2 = openfhe.fast_rotation %cc, %input1, %c2, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !openfhe.digit_decomp) -> !ct
    %res3 = openfhe.fast_rotation %cc, %input1, %c3, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !openfhe.digit_decomp) -> !ct
    %res4 = openfhe.fast_rotation %cc, %input1, %c4, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !openfhe.digit_decomp) -> !ct
    %sum1 = openfhe.add %cc, %res1, %res2 : (!cc, !ct, !ct) -> !ct
    %sum2 = openfhe.add %cc, %sum1, %res3 : (!cc, !ct, !ct) -> !ct
    %sum3 = openfhe.add %cc, %sum2, %res4 : (!cc, !ct, !ct) -> !ct
    return %sum3 : !ct
  }
}
