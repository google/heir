// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=simple_sum %s | FileCheck %s

!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi64>
  %0 = openfhe.rot %arg0, %arg1 {index = 16 : index} : (!openfhe.crypto_context, !ct) -> !ct
  %1 = openfhe.add %arg0, %arg1, %0 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %2 = openfhe.rot %arg0, %1 {index = 8 : index} : (!openfhe.crypto_context, !ct) -> !ct
  %3 = openfhe.add %arg0, %1, %2 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %4 = openfhe.rot %arg0, %3 {index = 4 : index} : (!openfhe.crypto_context, !ct) -> !ct
  %5 = openfhe.add %arg0, %3, %4 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %6 = openfhe.rot %arg0, %5 {index = 2 : index} : (!openfhe.crypto_context, !ct) -> !ct
  %7 = openfhe.add %arg0, %5, %6 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %8 = openfhe.rot %arg0, %7 {index = 1 : index} : (!openfhe.crypto_context, !ct) -> !ct
  %9 = openfhe.add %arg0, %7, %8 : (!openfhe.crypto_context, !ct, !ct) -> !ct
  %10 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<32xi64>) -> !pt
  %11 = openfhe.mul_plain %arg0, %9, %10 : (!openfhe.crypto_context, !ct, !pt) -> !ct
  %12 = openfhe.rot %arg0, %11 {index = 31 : index} : (!openfhe.crypto_context, !ct) -> !ct
  %14 = openfhe.mod_reduce %arg0, %12 : (!openfhe.crypto_context, !ct) -> !ct
  return %14 : !ct
}

// CHECK: @simple_sum
// CHECK: @simple_sum__generate_crypto_context
// CHECK: mulDepth = 1

// CHECK: @simple_sum__configure_crypto_context
// CHECK: openfhe.gen_rotkey
