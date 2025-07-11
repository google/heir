// High level source MLIR for CKKS (float, secret annotation), matching emit_simfhe.mlir after lowering
// RUN: bazel run //tools:heir-opt -- --mlir-to-ckks %s

!t = tensor<1024xf32>

module {
  func.func @test_ops(
      %x: !t {secret.secret},
      %y: !t {secret.secret},
      %z: !t
    ) -> (!t,!t,!t,!t,!t,!t,!t,!t) {
    %negate = arith.negf %x : !t
    %add = arith.addf %x, %y : !t
    %sub = arith.subf %x, %y : !t
    %mul1 = arith.mulf %x, %y : !t
    %mul2 = arith.mulf %mul1, %x : !t
    %c4 = arith.constant 4 : index
    %rot = tensor_ext.rotate %x, %c4 : !t, index
    %add_plain = arith.addf %x, %z : !t
    %sub_plain = arith.subf %x, %z : !t
    %mul_plain = arith.mulf %x, %z : !t
    return %negate, %add, %sub, %mul2, %rot, %add_plain, %sub_plain, %mul_plain :
       !t, !t, !t, !t, !t, !t, !t, !t
  }
}
