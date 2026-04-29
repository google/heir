// RUN: heir-opt %s | FileCheck %s

// CHECK: ![[RK5_TYPE:.*]] = !key_mgmt.rot_key<rotation_index = 5>
!rk5 = !key_mgmt.rot_key<rotation_index = 5>
// CHECK: ![[RK_DYN_TYPE:.*]] = !key_mgmt.rot_key<>
!rk_dynamic = !key_mgmt.rot_key<>

module {
  // CHECK: func.func @test_syntax
  func.func @test_syntax(%index: i64, %ct: i32) -> i32 {
    // CHECK: %[[C5:.*]] = arith.constant 5 : i64
    %c5 = arith.constant 5 : i64

    // CHECK: key_mgmt.prefetch_key %[[C5]]
    key_mgmt.prefetch_key %c5

    // CHECK: %[[RK_SSA:.*]] = key_mgmt.load_key %[[C5]] : i64 -> ![[RK5_TYPE]]
    %rk = key_mgmt.load_key %c5 : i64 -> !rk5

    // CHECK: %[[RK_DYN_SSA:.*]] = key_mgmt.load_key %arg0 : i64 -> ![[RK_DYN_TYPE]]
    %rk_dyn = key_mgmt.load_key %index : i64 -> !rk_dynamic

    // CHECK: %[[RK_USE_SSA:.*]] = key_mgmt.use_key %[[RK_SSA]] : ![[RK5_TYPE]] -> ![[RK_DYN_TYPE]]
    %rk_use = key_mgmt.use_key %rk : !rk5 -> !rk_dynamic

    // CHECK: %[[RK_ASSUME_SSA:.*]] = key_mgmt.assume_loaded %[[C5]] : i64 -> ![[RK5_TYPE]]
    %rk_assume = key_mgmt.assume_loaded %c5 : i64 -> !rk5

    // CHECK: key_mgmt.clear_key %[[RK_SSA]] : ![[RK5_TYPE]]
    key_mgmt.clear_key %rk : !rk5

    // CHECK: return %arg1
    return %ct : i32
  }
}
