// RUN: heir-opt --secret-insert-mgmt-ckks=include-first-mul=false --generate-param-ckks --secret-distribute-generic --secret-to-ckks %s | FileCheck %s

module {
// CHECK-LABEL: func @hv_matmul
// CHECK-SAME:    %[[ct:.*]]: !ct_L1_
// CHECK:      %[[pt:.*]] = lwe.rlwe_encode
// CHECK-NEXT: %[[ct2:.*]] = ckks.mul_plain %[[ct]], %[[pt]]
// CHECK:      %[[pt3:.*]] = lwe.rlwe_encode
// CHECK-NEXT: %[[ct4:.*]] = ckks.add_plain %[[ct2]], %[[pt3]]
// CHECK:      %[[ct5:.*]], %[[ct6:.*]] = affine.for %[[arg0:.*]] = 1 to 1024 iter_args(%[[ct8:.*]] = %[[ct4]], %[[ct9:.*]] = %[[ct]])
// CHECK-NEXT:   %[[ct10:.*]] = ckks.rotate %[[ct9]] {offset = 1
// CHECK-NEXT:   %[[extracted_slice:.*]] = tensor.extract_slice
// CHECK-NEXT:   %[[pt11:.*]] = lwe.rlwe_encode %[[extracted_slice]]
// CHECK-NEXT:   %[[ct12:.*]] = ckks.mul_plain %[[ct10]], %[[pt11]]
// CHECK-NEXT:   %[[ct13:.*]] = ckks.add %[[ct8]], %[[ct12]]
// CHECK-NEXT:   affine.yield %[[ct13]], %[[ct10]]
// CHECK:      %[[ct7:.*]] = ckks.rescale %[[ct5]]
// CHECK-NEXT: return %[[ct7]]
  func.func @hv_matmul(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = arith.mulf %input0, %cst : tensor<1x1024xf32>
      %2 = arith.addf %1, %cst_1 : tensor<1x1024xf32>
      %3:2 = affine.for %arg1 = 1 to 1024 iter_args(%arg2 = %2, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %4 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %5 = arith.mulf %4, %extracted_slice : tensor<1x1024xf32>
        %6 = arith.addf %arg2, %5 : tensor<1x1024xf32>
        affine.yield %6, %4 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      secret.yield %3#0 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
