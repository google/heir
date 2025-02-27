// RUN: heir-opt --split-input-file --verify-diagnostics %s 2>&1

// -----
// CHECK-NOT: test_invalid_tensor_length
func.func @test_invalid_tensor_length(%arg0 : tensor<1024x!mod_arith.int<33538049:i32>>, %arg1 : tensor<1024x!mod_arith.int<33538049:i32>>) ->  tensor<1024x!mod_arith.int<33538049:i32>> {
    // expected-error@below {{'pisa.add' op operand #0 must be tensor<8192xmod_arith.int< ... : i32>>, but got 'tensor<1024x!mod_arith.int<33538049 : i32>>'}}
    %0 = pisa.add %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<1024x!mod_arith.int<33538049:i32>>
  return %0 : tensor<1024x!mod_arith.int<33538049:i32>>
}

// -----
// CHECK-NOT: test_invalid_tensor_modulus_type
func.func @test_invalid_tensor_modulus_type(%arg0 : tensor<8192x!mod_arith.int<33538049:i64>>, %arg1 : tensor<8192x!mod_arith.int<33538049:i64>>) ->  tensor<8192x!mod_arith.int<33538049:i64>> {
    // expected-error@below {{'pisa.add' op operand #0 must be tensor<8192xmod_arith.int< ... : i32>>, but got 'tensor<8192x!mod_arith.int<33538049 : i64>>'}}
    %0 = pisa.add %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!mod_arith.int<33538049:i64>>
  return %0 : tensor<8192x!mod_arith.int<33538049:i64>>
}

// -----
// CHECK-NOT: test_invalid_modulus
func.func @test_invalid_modulus(%arg0 : tensor<8192x!mod_arith.int<33538049:i32>>, %arg1 : tensor<8192x!mod_arith.int<33538049:i32>>) ->  tensor<8192x!mod_arith.int<33538049:i32>> {
    // expected-error@below {{custom op 'pisa.add' 'pisa.add' op attribute 'q' failed to satisfy constraint: 32-bit signless integer attribute}}
    %0 = pisa.add %arg0, %arg1 {q = 18446744073709551557, i = 0 : i32} : tensor<8192x!mod_arith.int<33538049:i32>>
  return %0 : tensor<8192x!mod_arith.int<33538049:i32>>
}
