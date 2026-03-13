// RUN: heir-opt %s | FileCheck %s

// CHECK: [[rk5:!.*]] = !kmrt.rot_key<rotation_index = 5>
!rk5 = !kmrt.rot_key<rotation_index = 5>
// CHECK: [[rk_dynamic:!.*]] = !kmrt.rot_key
!rk_dynamic = !kmrt.rot_key<>

module {
  // CHECK: func.func @test_syntax
  func.func @test_syntax(%index: i64, %ct: i32) -> i32 {
    // CHECK: %[[C5:.*]] = arith.constant 5 : i64
    %c5 = arith.constant 5 : i64

    // CHECK: kmrt.prefetch_key %[[C5]]
    kmrt.prefetch_key %c5

    // CHECK: %[[RK:.*]] = kmrt.load_key %[[C5]] : i64 -> [[rk5]]
    %rk = kmrt.load_key %c5 : i64 -> !rk5

    // CHECK: %[[RK_DYN:.*]] = kmrt.load_key %arg0 : i64 -> [[rk_dynamic]]
    %rk_dyn = kmrt.load_key %index : i64 -> !rk_dynamic

    // CHECK: %[[RK_USE:.*]] = kmrt.use_key %[[RK]] : [[rk5]] -> [[rk_dynamic]]
    %rk_use = kmrt.use_key %rk : !rk5 -> !rk_dynamic

    // CHECK: %[[RK_ASSUME:.*]] = kmrt.assume_loaded %[[C5]] : i64 -> [[rk5]]
    %rk_assume = kmrt.assume_loaded %c5 : i64 -> !rk5

    // CHECK: %[[ROT:.*]] = kmrt.rotation %arg1, %[[RK]] : i32, [[rk5]] -> i32
    %rot = kmrt.rotation %ct, %rk : i32, !rk5 -> i32

    // CHECK: kmrt.clear_key %[[RK]] : [[rk5]]
    kmrt.clear_key %rk : !rk5

    // CHECK: return %[[ROT]]
    return %rot : i32
  }
}
