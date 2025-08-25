// RUN: heir-opt --math-to-polynomial-approximation %s | FileCheck %s

// CHECK: @test_maximumf
func.func @test_maximumf(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: arith.maximumf
  // CHECK-NOT: polynomial.eval

  // CHECK-DAG: arith.constant dense<0.03831
  // CHECK-DAG: arith.constant dense<5.0{{0*}}e-01> : tensor<10xf32>
  // CHECK-DAG: arith.constant dense<0.9370
  // CHECK-DAG: arith.constant dense<-0.5062
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
