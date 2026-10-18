// RUN: heir-opt --openfhe-fast-rotation-precompute %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module {
  func.func @simple_sum(%cc: !cc, %ct: !ct) -> !ct {
    // CHECK: openfhe.fast_rotation_precompute
    // CHECK-COUNT-4: openfhe.fast_rotation
    // CHECK-NOT: openfhe.rot
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi64>
    %ct_0 = openfhe.rot %cc, %ct <static_shift = 16 : index> : (!cc, !ct) -> !ct
    %ct_1 = openfhe.add %cc, %ct, %ct_0 : (!cc, !ct, !ct) -> !ct
    %ct_2 = openfhe.rot %cc, %ct <static_shift = 8 : index> : (!cc, !ct) -> !ct
    %ct_3 = openfhe.add %cc, %ct_1, %ct_2 : (!cc, !ct, !ct) -> !ct
    %ct_4 = openfhe.rot %cc, %ct <static_shift = 5 : index> : (!cc, !ct) -> !ct
    %ct_5 = openfhe.add %cc, %ct_3, %ct_4 : (!cc, !ct, !ct) -> !ct
    %ct_6 = openfhe.rot %cc, %ct <static_shift = 12 : index> : (!cc, !ct) -> !ct
    %ct_7 = openfhe.add %cc, %ct_5, %ct_6 : (!cc, !ct, !ct) -> !ct
    return %ct_7 : !ct
  }

  // CHECK: func.func @loop_invariant_rotation
  // CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct)
  func.func @loop_invariant_rotation(%cc: !cc, %ct: !ct) -> !ct {
    // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute %[[CC]], %[[CT]]
    // CHECK: affine.for
    // CHECK: %[[FAST_ROT:.*]] = openfhe.fast_rotation %[[CC]], %[[CT]], %{{.*}}, %[[PRECOMP]]
    // CHECK-NOT: openfhe.rot
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %0 = affine.for %i = 0 to 10 iter_args(%sum_iter = %ct) -> !ct {
      %ct_rot = openfhe.rot %cc, %ct <static_shift = 1 : index> : (!cc, !ct) -> !ct
      %sum_next = openfhe.add %cc, %sum_iter, %ct_rot : (!cc, !ct, !ct) -> !ct
      affine.yield %sum_next : !ct
    }
    return %0 : !ct
  }

  // CHECK: func.func @dynamic_shift_loop
  // CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct)
  func.func @dynamic_shift_loop(%cc: !cc, %ct: !ct) -> !ct {
    // CHECK: %[[PRECOMP:.*]] = openfhe.fast_rotation_precompute %[[CC]], %[[CT]]
    // CHECK: affine.for %[[IV:[^ ]*]] =
    // CHECK: %[[FAST_ROT:.*]] = openfhe.fast_rotation %[[CC]], %[[CT]], %[[IV]], %[[PRECOMP]]
    // CHECK-NOT: openfhe.rot
    %0 = affine.for %i = 0 to 10 iter_args(%sum_iter = %ct) -> !ct {
      %ct_rot = openfhe.rot %cc, %ct, %i : (!cc, !ct, index) -> !ct
      %sum_next = openfhe.add %cc, %sum_iter, %ct_rot : (!cc, !ct, !ct) -> !ct
      affine.yield %sum_next : !ct
    }
    return %0 : !ct
  }
}
