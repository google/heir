// RUN: heir-opt --openfhe-alloc-to-inplace %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!params = !openfhe.cc_params
!pk = !openfhe.public_key
!pt = !openfhe.plaintext
!sk = !openfhe.private_key

// Extracted dot_product_8f just before alloc_to_inplace
module attributes {backend.openfhe, scheme.ckks} {
  // CHECK: @dot_product([[cc:%[^:]*]]:
  func.func @dot_product(%cc: !cc, %arg0: tensor<1x!ct>, %arg1: tensor<1x!ct>) -> !ct {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<1.0> : tensor<1x8192xf32>
    %cst_0 = arith.constant dense<0.0> : tensor<1x8192xf32>
    %cst_1 = arith.constant 1.000000e-01 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x8192xf32>
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %extracted_3 = tensor.extract %arg1[%c0] : tensor<1x!ct>
    // CHECK: mul_no_relin
    %ct = openfhe.mul_no_relin %cc, %extracted, %extracted_3 : (!cc, !ct, !ct) -> !ct
    // CHECK: [[relin_result:%[^ ]*]] = openfhe.relin_inplace
    // The relin result is used directly by a rot, but also for an add_plain.
    // This means that the add_plain cannot be converted to an in-place op.

    %ct_4 = openfhe.relin %cc, %ct : (!cc, !ct) -> !ct
    %0 = scf.for %arg2 = %c0_i32 to %c8192_i32 step %c1_i32 iter_args(%arg3 = %cst_2) -> (tensor<1x8192xf32>)  : i32 {
      %5 = arith.index_cast %arg2 : i32 to index
      %inserted_27 = tensor.insert %cst_1 into %arg3[%c0, %5] : tensor<1x8192xf32>
      scf.yield %inserted_27 : tensor<1x8192xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8192] [1, 1] : tensor<1x8192xf32> to tensor<8192xf32>
    %1 = arith.extf %extracted_slice : tensor<8192xf32> to tensor<8192xf64>

    // anchoring filecheck on the next op to test
    // CHECK: openfhe.make_ckks_packed_plaintext
    %pt = openfhe.make_ckks_packed_plaintext %cc, %1 : (!cc, tensor<8192xf64>) -> !pt

    // CHECK-NEXT: [[add_plain_result:%.*]] = openfhe.add_plain [[cc]], [[relin_result]],
    %ct_5 = openfhe.add_plain %cc, %ct_4, %pt : (!cc, !ct, !pt) -> !ct
    // CHECK-NEXT: openfhe.rot [[cc]], [[add_plain_result]] {index = 6
    // CHECK-NEXT: openfhe.rot [[cc]], [[relin_result]] {index = 7
    %ct_6 = openfhe.rot %cc, %ct_5 {index = 6 : index} : (!cc, !ct) -> !ct
    %ct_7 = openfhe.rot %cc, %ct_4 {index = 7 : index} : (!cc, !ct) -> !ct
    // CHECK-COUNT-2: openfhe.add_inplace
    %ct_8 = openfhe.add %cc, %ct_6, %ct_7 : (!cc, !ct, !ct) -> !ct
    %ct_9 = openfhe.add %cc, %ct_8, %ct_4 : (!cc, !ct, !ct) -> !ct
    return %ct_9 : !ct
  }
}
