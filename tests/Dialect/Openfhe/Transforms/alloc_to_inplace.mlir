// RUN: heir-opt --openfhe-alloc-to-inplace %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext

module {
  // CHECK: func.func @add
  func.func @add(%cc: !cc, %ct: !ct) -> !ct {
    // CHECK-COUNT-3: openfhe.add_inplace
    %ct_0 = openfhe.add %cc, %ct, %ct : (!cc, !ct, !ct) -> !ct
    %ct_1 = openfhe.add %cc, %ct_0, %ct_0 : (!cc, !ct, !ct) -> !ct
    %ct_2 = openfhe.add %cc, %ct_1, %ct_1 : (!cc, !ct, !ct) -> !ct
    return %ct_2 : !ct
  }

  // CHECK: func.func @add_plain_lhs
  func.func @add_plain_lhs(%cc: !cc, %ct: !ct, %pt: !pt) -> !ct {
    // CHECK-COUNT-3: openfhe.add_plain_inplace
    %ct_0 = openfhe.add_plain %cc, %ct, %pt : (!cc, !ct, !pt) -> !ct
    %ct_1 = openfhe.add_plain %cc, %ct_0, %pt : (!cc, !ct, !pt) -> !ct
    %ct_2 = openfhe.add_plain %cc, %ct_1, %pt : (!cc, !ct, !pt) -> !ct
    return %ct_2 : !ct
  }

  // CHECK: func.func @add_plain_rhs
  func.func @add_plain_rhs(%cc: !cc, %ct: !ct, %pt: !pt) -> !ct {
    // CHECK-COUNT-3: openfhe.add_plain_inplace
    %ct_0 = openfhe.add_plain %cc, %pt, %ct : (!cc, !pt, !ct) -> !ct
    %ct_1 = openfhe.add_plain %cc, %pt, %ct_0 : (!cc, !pt, !ct) -> !ct
    %ct_2 = openfhe.add_plain %cc, %pt, %ct_1 : (!cc, !pt, !ct) -> !ct
    return %ct_2 : !ct
  }

  // CHECK: func.func @level_reduce
  func.func @level_reduce(%cc : !cc, %ct : !ct) -> !ct {
    // CHECK: openfhe.level_reduce_inplace
    %0 = openfhe.level_reduce %cc, %ct : (!cc, !ct) -> !ct
    return %0 : !ct
  }

  // CHECK: func.func @mul_const
  func.func @mul_const(%cc: !cc, %ct: !ct, %const: i64) -> !ct {
    // CHECK: openfhe.mul_const_inplace
    %ct_0 = openfhe.mul_const %cc, %ct, %const : (!cc, !ct, i64) -> !ct
    return %ct_0 : !ct
  }

}
