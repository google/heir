// RUN: heir-opt --mlir-to-bgv='ciphertext-degree=8192' --bgv-to-lwe --lwe-to-polynomial --convert-elementwise-to-affine --full-loop-unroll --convert-tensor-to-scalars --polynomial-to-pisa %s

// FIXME: ADD FILECHECK
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
