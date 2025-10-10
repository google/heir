// RUN: heir-opt --secretize --annotate-module="backend=openfhe scheme=ckks" --secret-insert-mgmt-ckks=slot-number=1024 --generate-param-ckks --secret-distribute-generic --canonicalize --secret-to-ckks %s | FileCheck %s

module {
// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext
// CHECK: func @hv_matmul
// CHECK-SAME:    %[[ct:.*]]: tensor<1x![[ct_ty]]>
// CHECK:      %[[extracted:.*]] = tensor.extract %[[ct]]
// CHECK:      %[[pt:.*]] = lwe.rlwe_encode
// CHECK-NEXT: %[[ct2:.*]] = ckks.mul_plain %[[extracted]], %[[pt]]
// CHECK:      %[[pt3:.*]] = lwe.rlwe_encode
// CHECK-NEXT: %[[ct4:.*]] = ckks.add_plain %[[ct2]], %[[pt3]]
// CHECK:      %[[ct5:.*]], %[[ct6:.*]] = affine.for %[[arg0:.*]] = 1 to 1024 iter_args(%[[ct7:.*]] = %[[ct4]], %[[ct8:.*]] = %[[extracted]])
// CHECK-NEXT:   %[[ct9:.*]] = ckks.rotate %[[ct8]] {offset = 1
// CHECK-NEXT:   %[[extracted_slice:.*]] = tensor.extract_slice
// CHECK:        %[[pt10:.*]] = lwe.rlwe_encode %[[extracted_slice]]
// CHECK-NEXT:   %[[ct11:.*]] = ckks.mul_plain %[[ct9]], %[[pt10]]
// CHECK-NEXT:   %[[ct12:.*]] = ckks.add %[[ct7]], %[[ct11]]
// CHECK-NEXT:   affine.yield %[[ct12]], %[[ct9]]
// CHECK:      %[[empty:.*]] = tensor.empty
// CHECK:      %[[result:.*]] = tensor.insert %[[ct5]] into %[[empty]]
// CHECK:      return %[[result]]
  func.func @hv_matmul(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %empty = tensor.empty() : tensor<1x1024xf32>
    %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %collapsed = tensor.extract_slice %input0[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
      %1 = arith.mulf %collapsed, %cst : tensor<1024xf32>
      %2 = arith.addf %1, %cst_1 : tensor<1024xf32>
      %3:2 = affine.for %arg1 = 1 to 1024 iter_args(%arg2 = %2, %arg3 = %collapsed) -> (tensor<1024xf32>, tensor<1024xf32>) {
        %4 = tensor_ext.rotate %arg3, %c1 : tensor<1024xf32>, index
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1024xf32>
        %5 = arith.mulf %4, %extracted_slice : tensor<1024xf32>
        %6 = arith.addf %arg2, %5 : tensor<1024xf32>
        affine.yield %6, %4 : tensor<1024xf32>, tensor<1024xf32>
      }
      %expanded = tensor.insert_slice %3#0 into %empty[0, 0] [1, 1024] [1, 1] : tensor<1024xf32> into tensor<1x1024xf32>
      secret.yield %expanded : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
