// RUN: heir-opt %s | FileCheck %s

// Test that the cheddar dialect can be parsed and printed.

// --- Setup operations ---

// CHECK: @test_create_context
func.func @test_create_context(%params: !cheddar.parameter) -> !cheddar.context {
  // CHECK: cheddar.create_context
  %ctx = cheddar.create_context %params : (!cheddar.parameter) -> !cheddar.context
  return %ctx : !cheddar.context
}

// CHECK: @test_create_user_interface
func.func @test_create_user_interface(%ctx: !cheddar.context) -> !cheddar.user_interface {
  // CHECK: cheddar.create_user_interface
  %ui = cheddar.create_user_interface %ctx : (!cheddar.context) -> !cheddar.user_interface
  return %ui : !cheddar.user_interface
}

// CHECK: @test_get_encoder
func.func @test_get_encoder(%ctx: !cheddar.context) -> !cheddar.encoder {
  // CHECK: cheddar.get_encoder
  %enc = cheddar.get_encoder %ctx : (!cheddar.context) -> !cheddar.encoder
  return %enc : !cheddar.encoder
}

// CHECK: @test_get_evk_map
func.func @test_get_evk_map(%ui: !cheddar.user_interface) -> !cheddar.evk_map {
  // CHECK: cheddar.get_evk_map
  %evk = cheddar.get_evk_map %ui : (!cheddar.user_interface) -> !cheddar.evk_map
  return %evk : !cheddar.evk_map
}

// CHECK: @test_get_mult_key
func.func @test_get_mult_key(%ui: !cheddar.user_interface) -> !cheddar.eval_key {
  // CHECK: cheddar.get_mult_key
  %key = cheddar.get_mult_key %ui : (!cheddar.user_interface) -> !cheddar.eval_key
  return %key : !cheddar.eval_key
}

// CHECK: @test_get_rot_key
func.func @test_get_rot_key(%ui: !cheddar.user_interface) -> !cheddar.eval_key {
  // CHECK: cheddar.get_rot_key
  // CHECK-SAME: distance = 5
  %key = cheddar.get_rot_key %ui {distance = 5 : i64} : (!cheddar.user_interface) -> !cheddar.eval_key
  return %key : !cheddar.eval_key
}

// CHECK: @test_get_conj_key
func.func @test_get_conj_key(%ui: !cheddar.user_interface) -> !cheddar.eval_key {
  // CHECK: cheddar.get_conj_key
  %key = cheddar.get_conj_key %ui : (!cheddar.user_interface) -> !cheddar.eval_key
  return %key : !cheddar.eval_key
}

// CHECK: @test_prepare_rot_key
func.func @test_prepare_rot_key(%ui: !cheddar.user_interface) {
  // CHECK: cheddar.prepare_rot_key
  // CHECK-SAME: distance = 3
  // CHECK-SAME: maxLevel = 10
  cheddar.prepare_rot_key %ui {distance = 3 : i64, maxLevel = 10 : i64} : (!cheddar.user_interface) -> ()
  return
}

// --- Encode / Encrypt / Decrypt ---

// CHECK: @test_encode
func.func @test_encode(
    %enc: !cheddar.encoder,
    %msg: tensor<4xf64>) -> !cheddar.plaintext {
  // CHECK: cheddar.encode
  // CHECK-SAME: level = 5
  // CHECK-SAME: scale = 45
  %pt = cheddar.encode %enc, %msg {level = 5 : i64, scale = 45 : i64} : (!cheddar.encoder, tensor<4xf64>) -> !cheddar.plaintext
  return %pt : !cheddar.plaintext
}

// CHECK: @test_encode_constant
func.func @test_encode_constant(
    %enc: !cheddar.encoder,
    %val: f64) -> !cheddar.constant {
  // CHECK: cheddar.encode_constant
  // CHECK-SAME: level = 3
  // CHECK-SAME: scale = 45
  %c = cheddar.encode_constant %enc, %val {level = 3 : i64, scale = 45 : i64} : (!cheddar.encoder, f64) -> !cheddar.constant
  return %c : !cheddar.constant
}

// CHECK: @test_decode
func.func @test_decode(
    %enc: !cheddar.encoder,
    %pt: !cheddar.plaintext) -> tensor<4xf64> {
  // CHECK: cheddar.decode
  %msg = cheddar.decode %enc, %pt : (!cheddar.encoder, !cheddar.plaintext) -> tensor<4xf64>
  return %msg : tensor<4xf64>
}

// CHECK: @test_encrypt
func.func @test_encrypt(
    %ui: !cheddar.user_interface,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: cheddar.encrypt
  %ct = cheddar.encrypt %ui, %pt : (!cheddar.user_interface, !cheddar.plaintext) -> !cheddar.ciphertext
  return %ct : !cheddar.ciphertext
}

// CHECK: @test_decrypt
func.func @test_decrypt(
    %ui: !cheddar.user_interface,
    %ct: !cheddar.ciphertext) -> !cheddar.plaintext {
  // CHECK: cheddar.decrypt
  %pt = cheddar.decrypt %ui, %ct : (!cheddar.user_interface, !cheddar.ciphertext) -> !cheddar.plaintext
  return %pt : !cheddar.plaintext
}

// --- Binary ct-ct operations ---

// CHECK: @test_add
func.func @test_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.add
  %result = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_sub
func.func @test_sub(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.sub
  %result = cheddar.sub %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_mult
func.func @test_mult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.mult
  %result = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Ct-pt / ct-const operations ---

// CHECK: @test_add_plain
func.func @test_add_plain(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: cheddar.add_plain
  %result = cheddar.add_plain %ctx, %ct, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_sub_plain
func.func @test_sub_plain(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: cheddar.sub_plain
  %result = cheddar.sub_plain %ctx, %ct, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_mult_plain
func.func @test_mult_plain(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: cheddar.mult_plain
  %result = cheddar.mult_plain %ctx, %ct, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_add_const
func.func @test_add_const(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %c: !cheddar.constant) -> !cheddar.ciphertext {
  // CHECK: cheddar.add_const
  %result = cheddar.add_const %ctx, %ct, %c : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_mult_const
func.func @test_mult_const(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %c: !cheddar.constant) -> !cheddar.ciphertext {
  // CHECK: cheddar.mult_const
  %result = cheddar.mult_const %ctx, %ct, %c : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Unary operations ---

// CHECK: @test_neg
func.func @test_neg(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.neg
  %result = cheddar.neg %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_rescale
func.func @test_rescale(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.rescale
  %result = cheddar.rescale %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_level_down
func.func @test_level_down(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.level_down
  // CHECK-SAME: targetLevel = 3
  %result = cheddar.level_down %ctx, %ct {targetLevel = 3 : i64} : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Key-switching operations ---

// CHECK: @test_relinearize
func.func @test_relinearize(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.relinearize
  %result = cheddar.relinearize %ctx, %ct, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_relinearize_rescale
func.func @test_relinearize_rescale(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.relinearize_rescale
  %result = cheddar.relinearize_rescale %ctx, %ct, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Fused compound operations ---

// CHECK: @test_hmult
func.func @test_hmult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hmult
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = true} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hmult_no_rescale
func.func @test_hmult_no_rescale(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = false} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hrot
func.func @test_hrot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hrot
  // CHECK-SAME: static_shift = 5
  %result = cheddar.hrot %ctx, %ct, %key {static_shift = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hrot_add
func.func @test_hrot_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  %result = cheddar.hrot_add %ctx, %ct0, %ct1, %key {distance = 3 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hconj
func.func @test_hconj(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hconj
  %result = cheddar.hconj %ctx, %ct, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hconj_add
func.func @test_hconj_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hconj_add
  %result = cheddar.hconj_add %ctx, %ct0, %ct1, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_mad_unsafe
func.func @test_mad_unsafe(
    %ctx: !cheddar.context,
    %acc: !cheddar.ciphertext,
    %ct: !cheddar.ciphertext,
    %c: !cheddar.constant) -> !cheddar.ciphertext {
  // CHECK: cheddar.mad_unsafe
  %result = cheddar.mad_unsafe %ctx, %acc, %ct, %c : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Extension operations ---

// CHECK: @test_boot
func.func @test_boot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // CHECK: cheddar.boot
  %result = cheddar.boot %ctx, %ct, %evk : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_linear_transform
func.func @test_linear_transform(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map,
    %diags: tensor<2x4xf64>) -> !cheddar.ciphertext {
  // CHECK: cheddar.linear_transform
  // CHECK-SAME: diagonal_indices = array<i32: 0, 1>
  // CHECK-SAME: level = 5
  // CHECK-SAME: logBabyStepGiantStepRatio = 0
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diags {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<2x4xf64>) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_eval_poly
func.func @test_eval_poly(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // CHECK: cheddar.eval_poly
  // CHECK-SAME: coefficients = [1.000000e+00, 2.000000e+00, 3.000000e+00]
  %result = cheddar.eval_poly %ctx, %ct, %evk {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}
