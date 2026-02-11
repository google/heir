// RUN: heir-opt --arith-to-mod-arith=modulus=65536 %s | FileCheck %s

// CHECK: @test_cast
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<10xf32>
// CHECK: return %[[CST]] : tensor<10xf32>
module {
  func.func @test_cast() -> tensor<10xf32> {
    %0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}
