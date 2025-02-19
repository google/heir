// This is effectively the pipeline run for this test, but it is run by bazel
// and not lit, so the official definition of what command is run can be found
// in the BUILD file for this directory, and the openfhe_end_to_end_test macro
// in test.bzl
//
// heir-opt --mlir-to-bgv='ciphertext-degree=32' --scheme-to-openfhe %s | bazel-bin/tools/heir-translate --emit-openfhe-pke

func.func @simple_sum(%arg0: tensor<32xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %i = 0 to 32 iter_args(%sum_iter = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%i] : tensor<32xi16>
    %2 = arith.addi %1, %sum_iter : i16
    affine.yield %2 : i16
  }
  return %0 : i16
}
