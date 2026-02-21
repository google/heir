// RUN: heir-opt --openfhe-hoist-key-switch-down %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!digit_decomp = !openfhe.digit_decomp

module {
  // CHECK: @test_hoist_add_two_rotations
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: %[[R1_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[R2_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[SUM:.*]] = openfhe.add %{{.*}}, %[[R1_EXT]], %[[R2_EXT]]
  // CHECK: %{{.*}} = openfhe.key_switch_down %{{.*}}, %[[SUM]]
  // CHECK-NOT: openfhe.key_switch_down{{.*}}openfhe.add
  // Typical pattern after convert-to-extended-basis: two rotations added together
  func.func @test_hoist_add_two_rotations(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r1 = openfhe.key_switch_down %cc, %r1_ext : (!cc, !ct) -> !ct
    %r2_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.key_switch_down %cc, %r2_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    return %sum : !ct
  }

  // CHECK: @test_hoist_mul_two_rotations
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: %[[R1_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[R2_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[PROD:.*]] = openfhe.mul %{{.*}}, %[[R1_EXT]], %[[R2_EXT]]
  // CHECK: %{{.*}} = openfhe.key_switch_down %{{.*}}, %[[PROD]]
  // CHECK-NOT: openfhe.key_switch_down{{.*}}openfhe.mul
  func.func @test_hoist_mul_two_rotations(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r1 = openfhe.key_switch_down %cc, %r1_ext : (!cc, !ct) -> !ct
    %r2_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.key_switch_down %cc, %r2_ext : (!cc, !ct) -> !ct
    %prod = openfhe.mul %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    return %prod : !ct
  }

  // CHECK: @test_hoist_add_inplace
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: %[[R1_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[R2_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[SUM:.*]] = openfhe.add_inplace %{{.*}}, %[[R1_EXT]], %[[R2_EXT]]
  // CHECK: %{{.*}} = openfhe.key_switch_down %{{.*}}, %[[SUM]]
  func.func @test_hoist_add_inplace(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r1 = openfhe.key_switch_down %cc, %r1_ext : (!cc, !ct) -> !ct
    %r2_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.key_switch_down %cc, %r2_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add_inplace %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    return %sum : !ct
  }

  // CHECK: @test_hoist_rotation_plus_ciphertext
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: %[[R_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: %[[SUM:.*]] = openfhe.add %{{.*}}, %[[R_EXT]], %{{.*}}
  // CHECK: %{{.*}} = openfhe.key_switch_down %{{.*}}, %[[SUM]]
  // Hoisting when only one operand is a rotation (common pattern)
  func.func @test_hoist_rotation_plus_ciphertext(%cc: !cc, %ct: !ct, %other: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r = openfhe.key_switch_down %cc, %r_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %r, %other : (!cc, !ct, !ct) -> !ct
    return %sum : !ct
  }

  // CHECK: @test_sum_of_multiple_rotations
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: %[[R1_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c1, %[[PRECOMP]]
  // CHECK: %[[R2_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c2, %[[PRECOMP]]
  // CHECK: %[[R3_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c3, %[[PRECOMP]]
  // CHECK: %[[R4_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c4, %[[PRECOMP]]
  // CHECK: %[[SUM1:.*]] = openfhe.add %{{.*}}, %[[R1_EXT]], %[[R2_EXT]]
  // CHECK: %[[SUM2:.*]] = openfhe.add %{{.*}}, %[[SUM1]], %[[R3_EXT]]
  // CHECK: %[[SUM3:.*]] = openfhe.add %{{.*}}, %[[SUM2]], %[[R4_EXT]]
  // CHECK: %{{.*}} = openfhe.key_switch_down %{{.*}}, %[[SUM3]]
  // CHECK-NOT: openfhe.key_switch_down{{.*}}openfhe.add
  // Realistic pattern: sum of multiple rotations (e.g., dot product)
  func.func @test_sum_of_multiple_rotations(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r1 = openfhe.key_switch_down %cc, %r1_ext : (!cc, !ct) -> !ct
    %r2_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.key_switch_down %cc, %r2_ext : (!cc, !ct) -> !ct
    %r3_ext = openfhe.fast_rotation_ext %cc, %ct, %c3, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r3 = openfhe.key_switch_down %cc, %r3_ext : (!cc, !ct) -> !ct
    %r4_ext = openfhe.fast_rotation_ext %cc, %ct, %c4, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r4 = openfhe.key_switch_down %cc, %r4_ext : (!cc, !ct) -> !ct
    %sum1 = openfhe.add %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    %sum2 = openfhe.add %cc, %sum1, %r3 : (!cc, !ct, !ct) -> !ct
    %sum3 = openfhe.add %cc, %sum2, %r4 : (!cc, !ct, !ct) -> !ct
    return %sum3 : !ct
  }

  // CHECK: @test_no_hoist_multiple_uses
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: %[[R1:.*]] = openfhe.key_switch_down
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.key_switch_down
  // CHECK: openfhe.add
  // No hoisting when key_switch_down result is used elsewhere
  func.func @test_no_hoist_multiple_uses(%cc: !cc, %ct: !ct) -> (!ct, !ct) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r1 = openfhe.key_switch_down %cc, %r1_ext : (!cc, !ct) -> !ct
    %r2_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.key_switch_down %cc, %r2_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    // %r1 is used twice, so no hoisting should occur
    return %sum, %r1 : !ct, !ct
  }

  // CHECK: @test_eliminate_redundant_key_switch
  // CHECK: openfhe.fast_rotation_ext
  // CHECK-NEXT: %[[KS:.*]] = openfhe.key_switch_down
  // CHECK-NEXT: return %[[KS]]
  // CHECK-NOT: openfhe.key_switch_down{{.*}}openfhe.key_switch_down
  // Eliminate redundant nested key_switch_down (edge case)
  func.func @test_eliminate_redundant_key_switch(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r = openfhe.key_switch_down %cc, %r_ext : (!cc, !ct) -> !ct
    %redundant = openfhe.key_switch_down %cc, %r : (!cc, !ct) -> !ct
    return %redundant : !ct
  }

  // CHECK: @test_mixed_operations
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: %[[R1_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c1, %[[PRECOMP]]
  // CHECK: %[[R2_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c2, %[[PRECOMP]]
  // CHECK: %[[R3_EXT:.*]] = openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %c3, %[[PRECOMP]]
  // CHECK: %[[SUM:.*]] = openfhe.add %{{.*}}, %[[R1_EXT]], %[[R2_EXT]]
  // CHECK: %[[PROD:.*]] = openfhe.mul %{{.*}}, %[[SUM]], %[[R3_EXT]]
  // CHECK: %{{.*}} = openfhe.key_switch_down %{{.*}}, %[[PROD]]
  // Mixed add and mul operations with rotations
  func.func @test_mixed_operations(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1_ext = openfhe.fast_rotation_ext %cc, %ct, %c1, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r1 = openfhe.key_switch_down %cc, %r1_ext : (!cc, !ct) -> !ct
    %r2_ext = openfhe.fast_rotation_ext %cc, %ct, %c2, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.key_switch_down %cc, %r2_ext : (!cc, !ct) -> !ct
    %r3_ext = openfhe.fast_rotation_ext %cc, %ct, %c3, %precomp : (!cc, !ct, index, !digit_decomp) -> !ct
    %r3 = openfhe.key_switch_down %cc, %r3_ext : (!cc, !ct) -> !ct
    %sum = openfhe.add %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    %prod = openfhe.mul %cc, %sum, %r3 : (!cc, !ct, !ct) -> !ct
    return %prod : !ct
  }
}
