// RUN: heir-opt --debug-validate-names %s | FileCheck %s

// CHECK: module
module {
  func.func @test_valid(%arg0: i32) {
    // CHECK: debug.validate %{{.*}} {name = "val1"} : i32
    debug.validate %arg0 {name = "val1"} : i32
    // CHECK: debug.validate %{{.*}} {name = "val2"} : i32
    debug.validate %arg0 {name = "val2"} : i32
    return
  }
}
