// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
!m32 = !mod_arith.int<33538049:i32>

// CHECK-LABEL: test_padd
func.func @test_padd(%arg0 : tensor<8192x!m32>, %arg1 : tensor<8192x!m32>) ->  tensor<8192x!m32> {
    %0 = pisa.add %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
  return %0 : tensor<8192x!m32>
}

// CHECK-LABEL: test_psub
func.func @test_psub(%arg0 : tensor<8192x!m32>, %arg1 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
    %0 = pisa.sub %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
  return %0 : tensor<8192x!m32>
}

// CHECK-LABEL: test_pmul
func.func @test_pmul(%arg0 : tensor<8192x!m32>, %arg1 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
    %0 = pisa.mul %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
  return %0 : tensor<8192x!m32>
}

// CHECK-LABEL: test_pmuli
func.func @test_pmuli(%arg0 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
    %0 = pisa.muli %arg0 {q = 2147483647 : i32, i = 0 : i32, imm = 5 : i32} : tensor<8192x!m32>
  return %0 : tensor<8192x!m32>
}

// CHECK-LABEL: test_pmac
func.func @test_pmac(%arg0 : tensor<8192x!m32>, %arg1 : tensor<8192x!m32>, %arg2 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
    %0 = pisa.mac %arg0, %arg1, %arg2 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
  return %0 : tensor<8192x!m32>
}

// CHECK-LABEL: test_pmaci
func.func @test_pmaci(%arg0 : tensor<8192x!m32>, %arg1 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
    %0 = pisa.maci %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32, imm = 5 : i32} : tensor<8192x!m32>
  return %0 : tensor<8192x!m32>
}

// FIXME: re-enable check once mod_arith.constant works for tensors
// func.func @test_pntt(%arg0 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
//     //TODO: figure out how to best handle the twiddle factors here...
//     // FIXME: cannot currently create a mod_arith.constant tensor? Below will silently fail and cause mlir-opt to produce no output?
//     %w = mod_arith.constant 42 : tensor<8192x!m32>
//     %0 = pisa.ntt %arg0, %w {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
//   return %0 : tensor<8192x!m32>
// }

// FIXME: re-enable check once mod_arith.constant works for tensors
// func.func @test_pintt(%arg0 : tensor<8192x!m32>) ->  tensor<8192x!m32>  {
//     //TODO: figure out how to best handle the twiddle factors here...
//     //FIXME: cannot currently create a mod_arith.constant tensor? Below will silently fail and cause mlir-opt to produce no output?
//     %w = mod_arith.constant 42 : tensor<8192x!m32>
//     %0 = pisa.intt %arg0, %w {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
//   return %0 : tensor<8192x!m32>
// }
