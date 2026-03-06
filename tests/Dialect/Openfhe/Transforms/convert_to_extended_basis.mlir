// RUN: heir-opt --openfhe-convert-to-extended-basis %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!digit_decomp = !openfhe.digit_decomp

module {
  // CHECK: @test_convert_single_rotation
  // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute
  // CHECK: openfhe.fast_rotation_ext %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMP]]
  // CHECK: openfhe.key_switch_down
  // CHECK-NOT: openfhe.fast_rotation {{.*}}cyclotomicOrder
  func.func @test_convert_single_rotation(%cc: !cc, %ct: !ct) -> !ct {
    %c4 = arith.constant 4 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %rot = openfhe.fast_rotation %cc, %ct, %c4, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !digit_decomp) -> !ct
    return %rot : !ct
  }

  // CHECK: @test_convert_multiple_rotations
  // CHECK: openfhe.fast_rotation_precompute
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.key_switch_down
  // CHECK: openfhe.fast_rotation_ext
  // CHECK: openfhe.key_switch_down
  // CHECK-NOT: openfhe.fast_rotation {{.*}}cyclotomicOrder
  func.func @test_convert_multiple_rotations(%cc: !cc, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %precomp = openfhe.fast_rotation_precompute %cc, %ct : (!cc, !ct) -> !digit_decomp
    %r1 = openfhe.fast_rotation %cc, %ct, %c1, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !digit_decomp) -> !ct
    %r2 = openfhe.fast_rotation %cc, %ct, %c2, %precomp {cyclotomicOrder = 64 : index} : (!cc, !ct, index, !digit_decomp) -> !ct
    %sum = openfhe.add %cc, %r1, %r2 : (!cc, !ct, !ct) -> !ct
    return %sum : !ct
  }
}
