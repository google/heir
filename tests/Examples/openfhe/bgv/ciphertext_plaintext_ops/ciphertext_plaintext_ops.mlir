// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

// [(input1 + input2) - input3] * input4
module attributes {scheme.bgv} {
  func.func @test_ciphertext_plaintext_ops(%cc : !cc, %input1 : !ct, %input2 : !pt, %input3 : !pt, %input4 : !pt) -> !ct {
    %add_res = openfhe.add_plain %cc, %input1, %input2 : (!cc, !ct, !pt) -> !ct
    %sub_res = openfhe.sub_plain %cc, %add_res, %input3 : (!cc, !ct, !pt) -> !ct
    %mul_res = openfhe.mul_plain %cc, %sub_res, %input4 : (!cc, !ct, !pt) -> !ct
    return %mul_res : !ct
  }
}
