// RUN: heir-opt --math-to-polynomial-approximation %s | FileCheck %s

// CHECK: @test_maximumf
func.func @test_maximumf(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: arith.maximumf
  // CHECK-NOT: polynomial.eval

  // CHECK-DAG: arith.constant dense<-0.12656936
  // CHECK-DAG: arith.constant dense<2.0{{0*}}e+00> : tensor<10xf32>
  // CHECK-DAG: arith.constant dense<0.27865994
  // CHECK-DAG: arith.constant dense<0.316969961
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: return
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}
