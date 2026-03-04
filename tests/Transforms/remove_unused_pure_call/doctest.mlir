// RUN: heir-opt --remove-unused-pure-call %s | FileCheck %s

module {
  func.func @pure_func(%arg0: i32) -> i32 attributes {client.pack_func} {
    return %arg0 : i32
  }

  // CHECK: func @test_remove_unused
  // CHECK-SAME: (%[[ARG0:.*]]: i32)
  func.func @test_remove_unused(%arg0: i32) {
    // CHECK-NOT: call @pure_func
    %0 = func.call @pure_func(%arg0) : (i32) -> i32
    return
  }
}
