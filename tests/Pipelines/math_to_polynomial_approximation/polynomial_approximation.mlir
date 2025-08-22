// RUN: heir-opt --math-to-polynomial-approximation %s | FileCheck %s

// CHECK: @test_maximumf
func.func @test_maximumf(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: arith.maximumf
  // CHECK-NOT: polynomial.eval

  // CHECK: arith.constant dense<0.0383100063> : tensor<10xf32>
  // CHECK: arith.constant dense<5.000000e-01> : tensor<10xf32>
  // CHECK: arith.constant dense<0.93702799> : tensor<10xf32>
  // CHECK: arith.constant dense<2.57697515E-17> : tensor<10xf32>
  // CHECK: arith.constant dense<-0.506277442> : tensor<10xf32>
  // CHECK: arith.constant dense<-1.37120354E-16> : tensor<10xf32>
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: return
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}
