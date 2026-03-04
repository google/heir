// RUN: heir-opt --remove-unused-pure-call %s | FileCheck %s

// CHECK: module
module {
  func.func @pure_func(%arg0: i32) -> i32 attributes {client.pack_func} {
    return %arg0 : i32
  }

  func.func @not_pure_func(%arg0: i32) -> i32 {
    return %arg0 : i32
  }

  func.func @pure_multi_res(%arg0: i32) -> (i32, i32) attributes {client.pack_func} {
    return %arg0, %arg0 : i32, i32
  }

  // CHECK: func @test_remove_unused
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_remove_unused(%arg0: i32) {
    // CHECK-NOT: call @pure_func
    %0 = func.call @pure_func(%arg0) : (i32) -> i32
    return
  }

  // CHECK: func @test_remove_multi_unused
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_remove_multi_unused(%arg0: i32) {
    // CHECK-NOT: call @pure_multi_res
    %0, %1 = func.call @pure_multi_res(%arg0) : (i32) -> (i32, i32)
    return
  }

  // CHECK: func @test_keep_used
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_keep_used(%arg0: i32) -> i32 {
    // CHECK: %[[RES:.*]] = call @pure_func(%[[ARG0]])
    // CHECK: return %[[RES]]
    %0 = func.call @pure_func(%arg0) : (i32) -> i32
    return %0 : i32
  }

  // CHECK: func @test_keep_multi_used
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_keep_multi_used(%arg0: i32) -> i32 {
    // CHECK: %[[RES:.*]]:2 = call @pure_multi_res(%[[ARG0]])
    // CHECK: return %[[RES]]#0
    %0, %1 = func.call @pure_multi_res(%arg0) : (i32) -> (i32, i32)
    return %0 : i32
  }

  // CHECK: func @test_keep_not_pure
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_keep_not_pure(%arg0: i32) {
    // CHECK: call @not_pure_func(%[[ARG0]])
    %0 = func.call @not_pure_func(%arg0) : (i32) -> i32
    return
  }
}
