// RUN: heir-opt --mlir-to-ckks=ckks-bootstrap-waterline=3 --scheme-to-openfhe %s | FileCheck %s

// CHECK: func.func @bootstrap_waterline
// CHECK:   openfhe.bootstrap

func.func @bootstrap_waterline(
    %x : f16 {secret.secret}
  ) -> f16 {
    %0 = arith.mulf %x, %x : f16
    %1 = arith.mulf %0, %0 : f16
    %2 = arith.mulf %1, %1 : f16
    %3 = arith.mulf %2, %2 : f16
    %4 = arith.mulf %3, %3 : f16
    %5 = arith.mulf %4, %4 : f16
    // cross level op
    %mixed0 = arith.mulf %5, %x : f16
  return %mixed0 : f16
}
