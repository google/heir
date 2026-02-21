// RUN: heir-opt --openfhe-fast-rotation-precompute %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module {
  func.func @simple_sum(%cc: !cc, %ct: !ct) -> !ct {
    // CHECK: openfhe.fast_rotation_precompute
    // CHECK-COUNT-4: openfhe.fast_rotation
    // CHECK-NOT: openfhe.rot
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi64>
    %ct_0 = openfhe.rot %cc, %ct {static_shift = 16 : index} : (!cc, !ct) -> !ct
    %ct_1 = openfhe.add %cc, %ct, %ct_0 : (!cc, !ct, !ct) -> !ct
    %ct_2 = openfhe.rot %cc, %ct {static_shift = 8 : index} : (!cc, !ct) -> !ct
    %ct_3 = openfhe.add %cc, %ct_1, %ct_2 : (!cc, !ct, !ct) -> !ct
    %ct_4 = openfhe.rot %cc, %ct {static_shift = 5 : index} : (!cc, !ct) -> !ct
    %ct_5 = openfhe.add %cc, %ct_3, %ct_4 : (!cc, !ct, !ct) -> !ct
    %ct_6 = openfhe.rot %cc, %ct {static_shift = 12 : index} : (!cc, !ct) -> !ct
    %ct_7 = openfhe.add %cc, %ct_5, %ct_6 : (!cc, !ct, !ct) -> !ct
    return %ct_7 : !ct
  }
}
