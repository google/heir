// RUN: heir-opt --secret-insert-mgmt-ckks=bootstrap-waterline=2 %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = f16, layout = #layout>
module {
  // CHECK: func @bootstrap_waterline
  // CHECK: %[[generic:.*]] = secret.generic
  // CHECK-NOT: mgmt.bootstrap {{.*}} !secret.secret
  // CHECK: return
  func.func @bootstrap_waterline(%arg0: !secret.secret<tensor<1x1024xf16>> {tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x1024xf16>> {tensor_ext.original_type = #original_type}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf16>>) {
    ^body(%input0: tensor<1x1024xf16>):
      %1 = arith.mulf %input0, %input0 : tensor<1x1024xf16>
      secret.yield %1 : tensor<1x1024xf16>
    } -> !secret.secret<tensor<1x1024xf16>>
    %r0 = mgmt.modreduce %0 : !secret.secret<tensor<1x1024xf16>>
    %r1 = mgmt.modreduce %r0 : !secret.secret<tensor<1x1024xf16>>
    %r2 = mgmt.modreduce %r1 : !secret.secret<tensor<1x1024xf16>>
    return %r2 : !secret.secret<tensor<1x1024xf16>>
  }
}
