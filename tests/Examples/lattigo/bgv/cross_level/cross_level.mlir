module attributes {backend.lattigo, bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [134250497, 17179967489, 17180262401, 68720050177], P = [68720066561, 68720295937], plaintextModulus = 65537, encryptionTechnique = extended>, scheme.bgv} {
  func.func @cross_level(%base: tensor<4xi16> {secret.secret}, %add: tensor<4xi16> {secret.secret}) -> tensor<4xi16> {
    // same level
    %base0 = arith.addi %base, %add : tensor<4xi16>
    // increase one level
    %mul1 = arith.muli %base0, %base0 : tensor<4xi16>
    // cross level add
    %base1 = arith.addi %mul1, %add : tensor<4xi16>
    // increase one level
    %mul2 = arith.muli %base1, %base1 : tensor<4xi16>
    // cross level add
    %base2 = arith.addi %mul2, %add : tensor<4xi16>
    // increase one level
    %mul3 = arith.muli %base2, %base2 : tensor<4xi16>
    // cross level add
    %base3 = arith.addi %mul3, %add : tensor<4xi16>
    return %base3 : tensor<4xi16>
  }
}
