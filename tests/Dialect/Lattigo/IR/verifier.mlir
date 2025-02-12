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

// -----

!rk = !lattigo.rlwe.relinearization_key
!gk = !lattigo.rlwe.galois_key<galoisElement = 5>
!ekset = !lattigo.rlwe.evaluation_key_set

func.func @test_rlwe_new_evaluation_key_set_operands_order(%rk: !rk, %gk: !gk) {
  // expected-error@+1 {{RLWERelinearizationKey must be the first key}}
  %ekset = lattigo.rlwe.new_evaluation_key_set %gk, %rk : (!gk, !rk) -> !ekset
  return
}

// -----

!ekset = !lattigo.rlwe.evaluation_key_set

func.func @test_rlwe_new_evaluation_key_set_operands_empty() {
  // expected-error@+1 {{must have at least one key}}
  %ekset = lattigo.rlwe.new_evaluation_key_set : () -> !ekset
  return
}

// -----

!gk = !lattigo.rlwe.galois_key<galoisElement = 5>
!ekset = !lattigo.rlwe.evaluation_key_set

func.func @test_rlwe_new_evaluation_key_set_operands_other_types(%gk: !gk) {
  %c = arith.constant 42 : i32
  // expected-error@+1 {{key must be of type RLWEGaloisKey}}
  %ekset = lattigo.rlwe.new_evaluation_key_set %gk, %c : (!gk, i32) -> !ekset
  return
}

// -----

!params = !lattigo.bgv.parameter
!encryptor_pk = !lattigo.rlwe.encryptor<publicKey = true>
!sk = !lattigo.rlwe.secret_key

func.func @test_rlwe_new_encryptor_pk_input_sk(%params: !params, %sk: !sk) {
  // expected-error@+1 {{encryption key and encryptor must have the same public/secret type}}
  %encryptor = lattigo.rlwe.new_encryptor %params, %sk : (!params, !sk) -> !encryptor_pk
  return
}

// -----

!params = !lattigo.bgv.parameter
!encryptor_sk = !lattigo.rlwe.encryptor<publicKey = false>
!pk = !lattigo.rlwe.public_key

func.func @test_rlwe_new_encryptor_pk_input_sk(%params: !params, %pk: !pk) {
  // expected-error@+1 {{encryption key and encryptor must have the same public/secret type}}
  %encryptor = lattigo.rlwe.new_encryptor %params, %pk : (!params, !pk) -> !encryptor_sk
  return
}
