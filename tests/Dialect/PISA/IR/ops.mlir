// RUN: not heir-opt --verify-diagnostics --split-input-file %s 2>&1 | FileCheck %s

// This simply tests for syntax.

// CHECK-LABEL: test_padd
func.func @test_padd(%arg0 : tensor<8192xi32>, %arg1 : tensor<8192xi32>) ->  tensor<8192xi32> {
    %0 = pisa.add %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_psub
func.func @test_psub(%arg0 : tensor<8192xi32>, %arg1 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    %0 = pisa.sub %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_pmul
func.func @test_pmul(%arg0 : tensor<8192xi32>, %arg1 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    %0 = pisa.mul %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_pmuli
func.func @test_pmuli(%arg0 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    %0 = pisa.muli %arg0 {q = 2147483647 : i32, i = 0 : i32, imm = 5 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_pmac
func.func @test_pmac(%arg0 : tensor<8192xi32>, %arg1 : tensor<8192xi32>, %arg2 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    %0 = pisa.mac %arg0, %arg1, %arg2 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_pmaci
func.func @test_pmaci(%arg0 : tensor<8192xi32>, %arg1 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    %0 = pisa.maci %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32, imm = 5 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_pntt
func.func @test_pntt(%arg0 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    //TODO: figure out how to best handle the twiddle factors here...
    %w = arith.constant dense<42> : tensor<8192xi32>
    %0 = pisa.ntt %arg0, %w {q = 2147483647 : i32, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}

// CHECK-LABEL: test_pintt
func.func @test_pintt(%arg0 : tensor<8192xi32>) ->  tensor<8192xi32>  {
    //TODO: figure out how to best handle the twiddle factors here...
    %w = arith.constant dense<42> : tensor<8192xi32>
    %0 = pisa.intt %arg0, %w {q = 2147483647 : i32, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}


// -----
// CHECK-NOT: test_invalid_tensor
func.func @test_invalid_tensor(%arg0 : tensor<1024xi32>, %arg1 : tensor<1024xi32>) ->  tensor<1024xi32> {
    // expected-error@below {{custom op 'pisa.add' 'pisa.add' op operand #0 must be tensor<8192xi32>, but got 'tensor<1024xi32>'}}
    %0 = pisa.add %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<1024xi32>
  return %0 : tensor<1024xi32>
}

// -----
// CHECK-NOT: test_invalid_modulus
func.func @test_invalid_modulus(%arg0 : tensor<8192xi32>, %arg1 : tensor<8192xi32>) ->  tensor<8192xi32> {
    // expected-error@below {{custom op 'pisa.add' 'pisa.add' op attribute 'q' failed to satisfy constraint: 32-bit signless integer attribute}}
    %0 = pisa.add %arg0, %arg1 {q = 18446744073709551557, i = 0 : i32} : tensor<8192xi32>
  return %0 : tensor<8192xi32>
}
