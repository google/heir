// RUN: heir-opt --math-to-polynomial-approximation %s | FileCheck %s

// CHECK: @test_maximumf
func.func @test_maximumf(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: arith.maximumf
  // CHECK-NOT: polynomial.eval

  // CHECK-DAG:  arith.constant dense<-0.06328
  // CHECK-DAG:  arith.constant dense<2.0000
  // CHECK-DAG:  arith.constant dense<0.2153
  // CHECK-DAG:  arith.constant dense<5.0000
  // CHECK-DAG:  arith.constant dense<0.3169
  // CHECK-DAG:  arith.constant dense<1.0000
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
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
