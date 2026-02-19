// RUN: heir-opt --secret-add-debug-port %s | FileCheck %s

module {
  func.func @test_lower_validate(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
    %0 = secret.generic(%arg0: !secret.secret<i32>) {
    ^body(%arg1: i32):
      debug.validate %arg1 {name = "val1"} : i32
      secret.yield %arg1 : i32
    } -> !secret.secret<i32>
    return %0 : !secret.secret<i32>
  }
}

// CHECK: func.func private @__heir_debug_i32(i32)
// CHECK-LABEL: func.func @test_lower_validate
// CHECK: call @__heir_debug_i32({{.*}}) {debug.name = "val1"}
