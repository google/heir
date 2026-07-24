// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks=bootstrap-waterline=3 %s | FileCheck %s

// CHECK: func.func @bootstrap_waterline
// CHECK: %[[generic:.*]] = secret.generic(%[[arg0:.*]]: !secret.secret<tensor<1x1024xf16>> {mgmt.mgmt = #mgmt.mgmt<level = 3>}) {
// CHECK:  (%[[input0:.*]]: tensor<1x1024xf16>):
// CHECK:    %[[v1:.*]] = arith.addf %[[input0]], %[[input0]] {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x1024xf16>
// CHECK:    %[[v2:.*]] = mgmt.modreduce %[[v1]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x1024xf16>
// CHECK:    %[[v3:.*]] = arith.addf %[[v2]], %[[v2]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x1024xf16>
// CHECK:    %[[v4:.*]] = mgmt.modreduce %[[v3]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf16>
// CHECK:    %[[v5:.*]] = arith.addf %[[v4]], %[[v4]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf16>
// CHECK:    %[[v6:.*]] = mgmt.modreduce %[[v5]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf16>
// CHECK:    %[[v7:.*]] = arith.addf %[[v6]], %[[v6]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf16>
// CHECK:    %[[boot:.*]] = mgmt.bootstrap %[[v7]] {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x1024xf16>
// CHECK:    %[[adj0:.*]] = mgmt.adjust_scale %[[boot]] {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x1024xf16>
// CHECK:    %[[v8:.*]] = mgmt.modreduce %[[adj0]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x1024xf16>
// CHECK:    %[[v9:.*]] = arith.addf %[[v8]], %[[v8]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x1024xf16>
// CHECK:    %[[v10:.*]] = mgmt.modreduce %[[v9]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf16>
// CHECK:    %[[v11:.*]] = arith.addf %[[v10]], %[[v10]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf16>
// CHECK:    %[[v12:.*]] = mgmt.level_reduce %[[input0]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x1024xf16>
// CHECK:    %[[v13:.*]] = mgmt.adjust_scale %[[v12]] {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x1024xf16>
// CHECK:    %[[v14:.*]] = mgmt.modreduce %[[v13]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf16>
// CHECK:    %[[v16:.*]] = arith.addf %[[v11]], %[[v14]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf16>
// CHECK:    secret.yield %[[v16]] : tensor<1x1024xf16>
// CHECK:  }



func.func @bootstrap_waterline(
    %x : f16 {secret.secret}
  ) -> f16 {
    %0 = arith.addf %x, %x : f16
    %r0 = mgmt.modreduce %0 : f16
    %1 = arith.addf %r0, %r0 : f16
    %r1 = mgmt.modreduce %1 : f16
    %2 = arith.addf %r1, %r1 : f16
    %r2 = mgmt.modreduce %2 : f16
    %3 = arith.addf %r2, %r2 : f16
    %r3 = mgmt.modreduce %3 : f16
    %4 = arith.addf %r3, %r3 : f16
    %r4 = mgmt.modreduce %4 : f16
    %5 = arith.addf %r4, %r4 : f16
    // cross level op
    %mixed0 = arith.addf %5, %x : f16
  return %mixed0 : f16
}
