// RUN: heir-opt --arith-to-cggi-quart  %s | FileCheck %s

// CHECK: return %[[RET:.*]] tensor<4x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
func.func @test_simple_split2(%arg0: i32, %arg1: i16) -> i32 {
  %2 = arith.constant 31 : i16
  %5 = arith.addi %arg1, %2 : i16
  %6 = arith.extui %5 : i16 to i32
  %7 = arith.addi %arg0, %6 : i32
  return %6 : i32
}
