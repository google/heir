// RUN: heir-opt --lattigo-configure-crypto-context %s | FileCheck %s

// The galois key for the rotation by -512 must be generated for galois
// element 5^(-512 mod 2N) mod 2N = 2049, not 1.
// CHECK: lattigo.rlwe.gen_galois_key
// CHECK-SAME: galoisElement = 2049

!ct = !lattigo.rlwe.ciphertext
!decryptor = !lattigo.rlwe.decryptor
!encoder = !lattigo.ckks.encoder
!encryptor_pk = !lattigo.rlwe.encryptor<publicKey = true>
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
!pt = !lattigo.rlwe.plaintext
module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.actual_slot_count = 4096 : i64, scheme.ckks, scheme.requested_slot_count = 1024 : i64} {
  func.func @rotate512__preprocessed(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %arg0: tensor<1x!ct>) -> tensor<1x!ct> attributes {client.preprocessed_func = {func_name = "rotate512"}} {
    %c0 = arith.constant 0 : index
    %c-512 = arith.constant -512 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %ct = lattigo.ckks.rotate_new %evaluator, %extracted, %c-512 : (!evaluator, !ct, index) -> !ct
    %0 = tensor.empty() : tensor<1x!ct>
    %inserted = tensor.insert %ct into %0[%c0] : tensor<1x!ct>
    return %inserted : tensor<1x!ct>
  }
  func.func @rotate512(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %arg0: tensor<1x!ct> {secret.secret}) -> (tensor<1x!ct> {secret.secret}) {
    %0 = call @rotate512__preprocessed(%evaluator, %param, %encoder, %arg0) : (!evaluator, !param, !encoder, tensor<1x!ct>) -> tensor<1x!ct>
    return %0 : tensor<1x!ct>
  }
  func.func @rotate512__encrypt__arg0(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %encryptor: !encryptor_pk, %arg0: tensor<1024xf32>) -> tensor<1x!ct> attributes {client.enc_func = {func_name = "rotate512", index = 0 : i64}} {
    %pt = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_0 = lattigo.ckks.encode %encoder, %arg0, %pt {scale = 45 : i64} : (!encoder, tensor<1024xf32>, !pt) -> !pt
    %ct = lattigo.rlwe.encrypt %encryptor, %pt_0 : (!encryptor_pk, !pt) -> !ct
    %from_elements = tensor.from_elements %ct : tensor<1x!ct>
    return %from_elements : tensor<1x!ct>
  }
  func.func @rotate512__decrypt__result0(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %decryptor: !decryptor, %arg0: tensor<1x!ct>) -> tensor<1024xf32> attributes {client.dec_func = {func_name = "rotate512", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %extracted = tensor.extract %arg0[%c0] : tensor<1x!ct>
    %pt = lattigo.rlwe.decrypt %decryptor, %extracted : (!decryptor, !ct) -> !pt
    %0 = lattigo.ckks.decode %encoder, %pt, %cst : (!encoder, !pt, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
