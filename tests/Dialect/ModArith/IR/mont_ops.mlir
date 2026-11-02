// RUN: heir-opt --mlir-print-local-scope %s | FileCheck %s

!Zp32 = !mod_arith.int<974258177 : i32>
!Zp32_tensor = tensor<4x!Zp32>

!Zp64 = !mod_arith.int<1125899908022273 : i64>
!Zp64_tensor = tensor<4x!Zp64>

// CHECK: func.func @test_mont_ops_32
func.func @test_mont_ops_32(%a: !Zp32, %b: !Zp32, %w: i64,
                            %at: !Zp32_tensor, %bt: !Zp32_tensor, %wt: tensor<4xi64>)
    -> (!Zp32, !Zp32, !Zp32, !Zp32, !Zp32, !Zp32, !Zp32_tensor) {
  // CHECK: mod_arith.mont_mul %{{.*}}, %{{.*}} : !mod_arith.int<974258177 : i32>
  %0 = mod_arith.mont_mul %a, %b : !Zp32

  // CHECK: mod_arith.mont_mul %{{.*}}, %{{.*}} lazy : !mod_arith.int<974258177 : i32>
  %1 = mod_arith.mont_mul %a, %b lazy : !Zp32

  // CHECK: mod_arith.mont_reduce %{{.*}} : i64 -> !mod_arith.int<974258177 : i32>
  %2 = mod_arith.mont_reduce %w : i64 -> !Zp32

  // CHECK: mod_arith.mont_reduce %{{.*}} lazy : i64 -> !mod_arith.int<974258177 : i32>
  %3 = mod_arith.mont_reduce %w lazy : i64 -> !Zp32

  // CHECK: mod_arith.to_mont %{{.*}} : !mod_arith.int<974258177 : i32>
  %4 = mod_arith.to_mont %a : !Zp32

  // CHECK: mod_arith.from_mont %{{.*}} : !mod_arith.int<974258177 : i32>
  %5 = mod_arith.from_mont %4 : !Zp32

  // CHECK: mod_arith.mont_mul %{{.*}}, %{{.*}} : tensor<4x!mod_arith.int<974258177 : i32>>
  %6 = mod_arith.mont_mul %at, %bt : !Zp32_tensor

  return %0, %1, %2, %3, %4, %5, %6 : !Zp32, !Zp32, !Zp32, !Zp32, !Zp32, !Zp32, !Zp32_tensor
}

// CHECK: func.func @test_mont_ops_64
func.func @test_mont_ops_64(%a: !Zp64, %b: !Zp64, %w: i128,
                            %at: !Zp64_tensor, %bt: !Zp64_tensor, %wt: tensor<4xi128>)
    -> (!Zp64, !Zp64, !Zp64, !Zp64, !Zp64, !Zp64, !Zp64_tensor) {
  // CHECK: mod_arith.mont_mul %{{.*}}, %{{.*}} : !mod_arith.int<1125899908022273 : i64>
  %0 = mod_arith.mont_mul %a, %b : !Zp64

  // CHECK: mod_arith.mont_mul %{{.*}}, %{{.*}} lazy : !mod_arith.int<1125899908022273 : i64>
  %1 = mod_arith.mont_mul %a, %b lazy : !Zp64

  // CHECK: mod_arith.mont_reduce %{{.*}} : i128 -> !mod_arith.int<1125899908022273 : i64>
  %2 = mod_arith.mont_reduce %w : i128 -> !Zp64

  // CHECK: mod_arith.mont_reduce %{{.*}} lazy : i128 -> !mod_arith.int<1125899908022273 : i64>
  %3 = mod_arith.mont_reduce %w lazy : i128 -> !Zp64

  // CHECK: mod_arith.to_mont %{{.*}} : !mod_arith.int<1125899908022273 : i64>
  %4 = mod_arith.to_mont %a : !Zp64

  // CHECK: mod_arith.from_mont %{{.*}} : !mod_arith.int<1125899908022273 : i64>
  %5 = mod_arith.from_mont %4 : !Zp64

  // CHECK: mod_arith.mont_mul %{{.*}}, %{{.*}} : tensor<4x!mod_arith.int<1125899908022273 : i64>>
  %6 = mod_arith.mont_mul %at, %bt : !Zp64_tensor

  return %0, %1, %2, %3, %4, %5, %6 : !Zp64, !Zp64, !Zp64, !Zp64, !Zp64, !Zp64, !Zp64_tensor
}
