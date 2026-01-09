!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext

module attributes {scheme.ckks} {
  func.func @simple_ckks_bootstrapping(%cc: !openfhe.crypto_context, %ct: !ct) -> !ct {
    %2 = openfhe.bootstrap %cc, %ct : (!openfhe.crypto_context, !ct) -> !ct
    return %2 : !ct
  }
}
