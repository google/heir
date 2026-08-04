// RUN: not heir-opt --secret-insert-mgmt-bgv="after-mul=true level-budget=5" %s 2>&1 | FileCheck %s --check-prefix=CHECK-BGV-FAIL
// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true level-budget=5 bootstrap-waterline=3" %s | FileCheck %s --check-prefix=CHECK-CKKS-PASS

// CHECK-BGV-FAIL: error: value has invalid level: %{{.*}} = "arith.muli"

// CHECK-CKKS-PASS: @test_loop
module {
  func.func @test_loop(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %0 = secret.generic(%arg0 : !secret.secret<i32>) {
    ^body(%val_sec: i32):
      %loop_res = scf.for %i = %c1 to %c10 step %c1 iter_args(%iter = %val_sec) -> (i32) {
        %next = arith.muli %iter, %val_sec : i32
        scf.yield %next : i32
      }
      %final = arith.muli %loop_res, %val_sec : i32
      secret.yield %final : i32
    } -> !secret.secret<i32>
    return %0 : !secret.secret<i32>
  }
}
