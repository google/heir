// This is effectively the pipeline run for this test, but it is run by bazel
// and not lit, so the official definition of what command is run can be found
// in the BUILD file for this directory, and the openfhe_end_to_end_test macro
// in test.bzl
//
// heir-opt --mlir-to-openfhe-bgv='entry-function=basic_test ciphertext-degree=8192' %s | bazel-bin/tools/heir-translate --emit-openfhe-pke

// Below is the pipeline used for pISA:
//  heir-opt --mlir-to-bgv='entry-function=basic_test ciphertext-degree=8192' --bgv-to-lwe --lwe-to-polynomial --convert-elementwise-to-affine --full-loop-unroll --convert-tensor-to-scalars polynomial-to-pisa %s | bazel-bin/tools/heir-translate --emit-pisa


!t = tensor<8192xi16>

func.func @basic_test(%x: !t {secret.secret}, %y: !t {secret.secret}) -> !t {
  %r0 = tensor.empty() : !t
  %r = affine.for %i = 0 to 8192 iter_args(%r = %r0) -> !t {
    %xi = tensor.extract %x[%i] : !t
    %yi = tensor.extract %y[%i] : !t
    %si = arith.addi %xi, %yi : i16
    %ri = tensor.insert %si into %r[%i] : !t
    affine.yield %ri : !t
  }
  return %r : !t
}
