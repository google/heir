// RUN: heir-opt --mlir-print-local-scope --secret-to-bgv %s | FileCheck %s

!secret_tensor_ty = !secret.secret<tensor<1x1024xi32>>
!extract_ty = !secret.secret<tensor<1024xi32>>
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // CHECK: func @test_extract_slice
  func.func @test_extract_slice(%arg0 : !secret_tensor_ty {mgmt.mgmt = #mgmt}) -> (!extract_ty {mgmt.mgmt = #mgmt}) {
    %0 = secret.generic(%arg0: !secret_tensor_ty) {
    // CHECK: tensor.extract
    // CHECK-NOT: slice
    ^body(%ARG0 : tensor<1x1024xi32>):
        %1 = tensor.extract_slice %ARG0[0, 0] [1, 1024] [1, 1] : tensor<1x1024xi32> to tensor<1024xi32>
        secret.yield %1 : tensor<1024xi32>
    } -> (!extract_ty {mgmt.mgmt = #mgmt})
    return %0 : !extract_ty
  }
}
