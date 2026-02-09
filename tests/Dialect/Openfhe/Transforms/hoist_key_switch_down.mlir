// RUN: heir-opt --openfhe-hoist-key-switch-down %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!digit_decomp = !openfhe.digit_decomp

module {
  // CHECK: @test_hoist_through_add
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.add
  // CHECK: openfhe.key_switch_down
  // CHECK-NOT: openfhe.key_switch_down{{.*}}openfhe.add
  func.func @test_hoist_through_add(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %a_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %b_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %a = openfhe.key_switch_down %cc, %a_ext : (!cc, !ct) -> !ct
    %b = openfhe.key_switch_down %cc, %b_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %a, %b : (!cc, !ct, !ct) -> !ct
    return %sum : !ct
  }

  // CHECK: @test_hoist_through_mul
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.mul
  // CHECK: openfhe.key_switch_down
  func.func @test_hoist_through_mul(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %a_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %b_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %a = openfhe.key_switch_down %cc, %a_ext : (!cc, !ct) -> !ct
    %b = openfhe.key_switch_down %cc, %b_ext : (!cc, !ct) -> !ct
    %prod = openfhe.mul %cc, %a, %b : (!cc, !ct, !ct) -> !ct
    return %prod : !ct
  }

  // CHECK: @test_no_hoist_single_ks
  // CHECK: openfhe.key_switch_down
  // CHECK: openfhe.add
  // No hoisting when only one operand is key_switch_down
  func.func @test_no_hoist_single_ks(%cc: !cc, %ct: !ct, %plain_ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %a_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %a = openfhe.key_switch_down %cc, %a_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %a, %plain_ct : (!cc, !ct, !ct) -> !ct
    return %sum : !ct
  }

  // CHECK: @test_no_hoist_multiple_uses
  // CHECK: openfhe.key_switch_down
  // CHECK: openfhe.key_switch_down
  // CHECK: openfhe.add
  // No hoisting when key_switch_down has multiple uses
  func.func @test_no_hoist_multiple_uses(%cc: !cc, %ct: !ct) -> (!ct, !ct) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %a_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %b_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %a = openfhe.key_switch_down %cc, %a_ext : (!cc, !ct) -> !ct
    %b = openfhe.key_switch_down %cc, %b_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %a, %b : (!cc, !ct, !ct) -> !ct
    // %a is used twice, so no hoisting
    return %sum, %a : !ct, !ct
  }
}
