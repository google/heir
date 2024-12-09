// RUN: heir-opt --verify-diagnostics --split-input-file %s

!pt = !lattigo.rlwe.plaintext

!encoder = !lattigo.bgv.encoder

!value = tensor<8xi32>
!value2 = tensor<9xi32>
!value_scalar = i32

func.func @test_bgv_encode_no_scalar(%encoder: !encoder, %value : !value_scalar, %pt: !pt) {
  // expected-error@+1 {{value must be a ranked tensor}}
  %encoded = lattigo.bgv.encode %encoder, %value, %pt : (!encoder, !value_scalar, !pt) -> !pt
  return
}

// -----

!pt = !lattigo.rlwe.plaintext

!encoder = !lattigo.bgv.encoder

!value = tensor<8xi32>
!value2 = tensor<9xi32>
!value_scalar = i32

func.func @test_bgv_decode_tensor_match(%encoder: !encoder, %pt: !pt, %value : !value) {
  // expected-error@+1 {{value and decoded types must match}}
  %decoded = lattigo.bgv.decode %encoder, %pt, %value : (!encoder, !pt, !value) -> !value2
  return
}

// -----

!pt = !lattigo.rlwe.plaintext

!encoder = !lattigo.bgv.encoder

!value = tensor<8xi32>
!value2 = tensor<9xi32>
!value_scalar = i32

func.func @test_bgv_decode_no_scalar(%encoder: !encoder, %pt: !pt, %value : !value_scalar) {
  // expected-error@+1 {{decoded must be a ranked tensor}}
  %decoded = lattigo.bgv.decode %encoder, %pt, %value : (!encoder, !pt, !value_scalar) -> !value_scalar
  return
}
