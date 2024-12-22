// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks=bootstrap-waterline=3 --mlir-to-openfhe-ckks %s | FileCheck %s

// CHECK: func.func @bootstrap_waterline
// CHECK:   openfhe.bootstrap

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
