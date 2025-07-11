// RUN: heir-opt --arith-to-cggi-quart  %s | FileCheck %s

// CHECK: ![[ct_ty:.*]] = !lwe.new_lwe_ciphertext

// CHECK: return %[[RET:.*]] tensor<4x![[ct_ty]]>
func.func @test_simple_split2(%arg0: i32, %arg1: i32) -> i32 {
  %2 = arith.constant 31 : i8
  %1 = arith.extui %2 : i8 to i32
  %5 = arith.addi %arg1, %1 : i32
  %7 = arith.muli %arg0, %5 : i32
  return %7 : i32
}
