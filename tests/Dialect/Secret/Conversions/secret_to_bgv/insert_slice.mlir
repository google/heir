// RUN: heir-opt --secret-to-bgv %s | FileCheck %s

!secret_tensor_ty = !secret.secret<tensor<1x1024xi32>>
!extract_ty = !secret.secret<tensor<1024xi32>>
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // CHECK: func @test_insert_slice
  func.func @test_insert_slice(%arg0 : !secret_tensor_ty {mgmt.mgmt = #mgmt}, %arg1: !extract_ty {mgmt.mgmt = #mgmt}) -> (!secret_tensor_ty {mgmt.mgmt = #mgmt}) {
    %0 = secret.generic(%arg0: !secret_tensor_ty, %arg1: !extract_ty) {
    // CHECK: tensor.insert
    // CHECK-NOT: slice
    ^body(%input0: tensor<1x1024xi32>, %input1: tensor<1024xi32>):
        %1 = tensor.insert_slice %input1 into %input0[0, 0] [1, 1024] [1, 1] : tensor<1024xi32> into tensor<1x1024xi32>
        secret.yield %1 : tensor<1x1024xi32>
    } -> (!secret_tensor_ty {mgmt.mgmt = #mgmt})
    return %0 : !secret_tensor_ty
  }
}
