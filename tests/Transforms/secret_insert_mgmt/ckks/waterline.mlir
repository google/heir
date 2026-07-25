// RUN: heir-opt --secret-insert-mgmt-ckks=bootstrap-waterline=4 %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = f16, layout = #layout>
module {
  // CHECK: func @bootstrap_waterline
  // CHECK-SAME: level = 4
  // CHECK-COUNT-4: arith.mulf
  // CHECK-NOT: arith.mulf
  // CHECK: mgmt.bootstrap
  // CHECK-COUNT-4: arith.mulf
  // CHECK-NOT: arith.mulf
  // CHECK: mgmt.bootstrap
  // CHECK-COUNT-3: arith.mulf
  // CHECK-NOT: arith.mulf
  // CHECK: secret.yield
  func.func @bootstrap_waterline(%arg0: !secret.secret<tensor<1x1024xf16>> {tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x1024xf16>> {tensor_ext.original_type = #original_type}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf16>>) {
    ^body(%input0: tensor<1x1024xf16>):
      %1 = arith.mulf %input0, %input0 : tensor<1x1024xf16>
      %2 = arith.mulf %1, %1 : tensor<1x1024xf16>
      %3 = arith.mulf %2, %2 : tensor<1x1024xf16>
      %4 = arith.mulf %3, %3 : tensor<1x1024xf16>
      %5 = arith.mulf %4, %4 : tensor<1x1024xf16>
      %6 = arith.mulf %5, %5 : tensor<1x1024xf16>
      %7 = arith.mulf %6, %6 : tensor<1x1024xf16>
      %8 = arith.mulf %7, %7 : tensor<1x1024xf16>
      %9 = arith.mulf %8, %8 : tensor<1x1024xf16>
      %10 = arith.mulf %9, %9 : tensor<1x1024xf16>
      %11 = arith.mulf %10, %10 : tensor<1x1024xf16>
      secret.yield %11 : tensor<1x1024xf16>
    } -> !secret.secret<tensor<1x1024xf16>>
    return %0 : !secret.secret<tensor<1x1024xf16>>
  }
}
