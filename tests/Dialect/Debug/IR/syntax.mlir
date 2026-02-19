// RUN: heir-opt %s | FileCheck %s

// CHECK: module
module {
  // CHECK: @test_syntax
  func.func @test_syntax(%arg0: i32) {
    // CHECK: debug.validate %arg0 {name = "test_val"} : i32
    debug.validate %arg0 { name = "test_val" } : i32
    // CHECK: debug.validate %arg0 {metadata = "{\22foo\22: 7}", name = "test_val_with_metadata"} : i32
    debug.validate %arg0 { name = "test_val_with_metadata", metadata = "{\"foo\": 7}" } : i32
    return
  }
}
