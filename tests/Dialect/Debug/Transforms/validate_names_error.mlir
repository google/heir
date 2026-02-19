// RUN: heir-opt --debug-validate-names --verify-diagnostics %s

module {
  func.func @test_duplicate_1(%arg0: i32) {
    debug.validate %arg0 {name = "val1"} : i32
    return
  }
  func.func @test_duplicate_2(%arg0: i32) {
    // expected-error@below {{duplicate debug.validate name: val1}}
    debug.validate %arg0 {name = "val1"} : i32
    return
  }
}
