// RUN: heir-opt --polynomial-approximation %s | FileCheck %s

// CHECK-LABEL: @test_fn
func.func @test_fn(%x: !secret.secret<i32>) {
    return
}
