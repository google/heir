// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {scheme.ckks} {
  func.func @test_binops(%cc : !cc, %input1 : !ct, %input2 : !ct) -> !ct {
    %add_res = openfhe.add %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
    %sub_res = openfhe.sub %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
    %mul_res = openfhe.mul %cc, %add_res, %sub_res : (!cc, !ct, !ct) -> !ct
    return %mul_res : !ct
  }
}
