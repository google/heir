// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks=bootstrap-waterline=3 %s | FileCheck %s

// CHECK: func.func @bootstrap_waterline(%arg0: !secret.secret<f16>) -> !secret.secret<f16> {
// CHECK: %0 = secret.generic ins(%[[arg0:.*]] : !secret.secret<f16>) attrs = {__argattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 3>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 1>}]} {
// CHECK:  (%[[input0:.*]]: f16):
// CHECK:    %[[v1:.*]] = arith.addf %[[input0]], %[[input0]] {mgmt.mgmt = #mgmt.mgmt<level = 3>} : f16
// CHECK:    %[[v2:.*]] = mgmt.modreduce %[[v1]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f16
// CHECK:    %[[v3:.*]] = arith.addf %2, %[[v2]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f16
// CHECK:    %[[v4:.*]] = mgmt.modreduce %[[v3]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f16
// CHECK:    %[[v5:.*]] = arith.addf %4, %[[v4]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f16
// CHECK:    %[[v6:.*]] = mgmt.modreduce %[[v5]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
// CHECK:    %[[v7:.*]] = mgmt.bootstrap %[[v6]] {mgmt.mgmt = #mgmt.mgmt<level = 3>} : f16
// CHECK:    %[[v8:.*]] = arith.addf %[[v7]], %[[v7]] {mgmt.mgmt = #mgmt.mgmt<level = 3>} : f16
// CHECK:    %[[v9:.*]] = mgmt.modreduce %[[v8]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f16
// CHECK:    %[[v10:.*]]  = arith.addf %[[v9]], %[[v9]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f16
// CHECK:    %[[v11:.*]]  = mgmt.modreduce %[[v10]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f16
// CHECK:    %[[v12:.*]]  = arith.addf %[[v11]], %[[v11]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f16
// CHECK:    %[[v13:.*]]  = mgmt.modreduce %[[input0]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f16
// CHECK:    %[[v14:.*]]  = mgmt.modreduce %[[v13]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f16
// CHECK:    %[[v15:.*]]  = arith.addf %[[v12]], %[[v14]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f16
// CHECK:    secret.yield %[[v15]] : f16


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
