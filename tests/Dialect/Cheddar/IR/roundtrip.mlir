// RUN: heir-opt %s | FileCheck %s

// Test that the cheddar dialect can be parsed and printed. Payload-producing
// ops are destination-passing: their payload operands are `tensor<!cheddar.X>`
// and they take a trailing `$output` init operand tied to the result.

// --- Setup operations ---

// CHECK: @test_make_parameter
func.func @test_make_parameter() -> !cheddar.parameter {
  // CHECK: cheddar.make_parameter
  // CHECK-SAME: logN = 14
  %p = cheddar.make_parameter {logN = 14 : i64, logScale = 45 : i64, mainPrimes = array<i64: 1, 2, 3>, auxPrimes = array<i64: 4, 5>} : !cheddar.parameter
  return %p : !cheddar.parameter
}

// CHECK: @test_create_context
func.func @test_create_context(%params: !cheddar.parameter, %init: tensor<!cheddar.context>) -> tensor<!cheddar.context> {
  // CHECK: cheddar.create_context
  %ctx = cheddar.create_context %params, %init : (!cheddar.parameter, tensor<!cheddar.context>) -> tensor<!cheddar.context>
  return %ctx : tensor<!cheddar.context>
}

// CHECK: @test_create_user_interface
func.func @test_create_user_interface(%ctx: tensor<!cheddar.context>, %init: tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface> {
  // CHECK: cheddar.create_user_interface
  %ui = cheddar.create_user_interface %ctx, %init : (tensor<!cheddar.context>, tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %ui : tensor<!cheddar.user_interface>
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

// CHECK: @test_prepare_rot_key
func.func @test_prepare_rot_key(%ui: tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface> {
  // CHECK: cheddar.prepare_rot_key
  // CHECK-SAME: distance = 3
  // CHECK-SAME: maxLevel = 10
  %ui2 = cheddar.prepare_rot_key %ui {distance = 3 : i64, maxLevel = 10 : i64} : (tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %ui2 : tensor<!cheddar.user_interface>
}

// --- Encode / Encrypt / Decrypt ---

// CHECK: @test_encode
func.func @test_encode(
    %enc: !cheddar.encoder,
    %msg: tensor<4xf64>,
    %out: tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext> {
  // CHECK: cheddar.encode
  // CHECK-SAME: level = 5
  // CHECK-SAME: scale = 0x42C0000000000000
  %pt = cheddar.encode %enc, %msg, %out {level = 5 : i64, scale = 35184372088832.0 : f64} : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  return %pt : tensor<!cheddar.plaintext>
}

// CHECK: @test_encode_constant
func.func @test_encode_constant(
    %enc: !cheddar.encoder,
    %val: f64,
    %out: tensor<!cheddar.constant>) -> tensor<!cheddar.constant> {
  // CHECK: cheddar.encode_constant
  // CHECK-SAME: level = 3
  // CHECK-SAME: scale = 0x42C0000000000000
  %c = cheddar.encode_constant %enc, %val, %out {level = 3 : i64, scale = 35184372088832.0 : f64} : (!cheddar.encoder, f64, tensor<!cheddar.constant>) -> tensor<!cheddar.constant>
  return %c : tensor<!cheddar.constant>
}

// CHECK: @test_decode
func.func @test_decode(
    %enc: !cheddar.encoder,
    %pt: tensor<!cheddar.plaintext>,
    %dst: tensor<4xf64>) -> tensor<4xf64> {
  // CHECK: cheddar.decode
  %msg = cheddar.decode %enc, %pt, %dst : (!cheddar.encoder, tensor<!cheddar.plaintext>, tensor<4xf64>) -> tensor<4xf64>
  return %msg : tensor<4xf64>
}

// CHECK: @test_encrypt
func.func @test_encrypt(
    %ui: !cheddar.user_interface,
    %pt: tensor<!cheddar.plaintext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.encrypt
  %ct = cheddar.encrypt %ui, %pt, %out : (!cheddar.user_interface, tensor<!cheddar.plaintext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %ct : tensor<!cheddar.ciphertext>
}

// CHECK: @test_decrypt
func.func @test_decrypt(
    %ui: !cheddar.user_interface,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext> {
  // CHECK: cheddar.decrypt
  %pt = cheddar.decrypt %ui, %ct, %out : (!cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  return %pt : tensor<!cheddar.plaintext>
}

// --- Binary ct-ct operations ---

// CHECK: @test_add
func.func @test_add(
    %ctx: !cheddar.context,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.add
  %result = cheddar.add %ctx, %ct0, %ct1, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_sub
func.func @test_sub(
    %ctx: !cheddar.context,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.sub
  %result = cheddar.sub %ctx, %ct0, %ct1, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_mult
func.func @test_mult(
    %ctx: !cheddar.context,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.mult
  %result = cheddar.mult %ctx, %ct0, %ct1, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// --- Ct-pt / ct-const operations ---

// CHECK: @test_add_plain
func.func @test_add_plain(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %pt: tensor<!cheddar.plaintext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.add_plain
  %result = cheddar.add_plain %ctx, %ct, %pt, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.plaintext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_sub_plain
func.func @test_sub_plain(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %pt: tensor<!cheddar.plaintext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.sub_plain
  %result = cheddar.sub_plain %ctx, %ct, %pt, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.plaintext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_mult_plain
func.func @test_mult_plain(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %pt: tensor<!cheddar.plaintext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.mult_plain
  %result = cheddar.mult_plain %ctx, %ct, %pt, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.plaintext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_add_const
func.func @test_add_const(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %c: tensor<!cheddar.constant>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.add_const
  %result = cheddar.add_const %ctx, %ct, %c, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.constant>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_mult_const
func.func @test_mult_const(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %c: tensor<!cheddar.constant>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.mult_const
  %result = cheddar.mult_const %ctx, %ct, %c, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.constant>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// --- Unary operations ---

// CHECK: @test_neg
func.func @test_neg(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.neg
  %result = cheddar.neg %ctx, %ct, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_rescale
func.func @test_rescale(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.rescale
  %result = cheddar.rescale %ctx, %ct, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_level_down
func.func @test_level_down(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.level_down
  // CHECK-SAME: targetLevel = 3
  %result = cheddar.level_down %ctx, %ct, %out {targetLevel = 3 : i64} : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// --- Key-switching operations ---

// CHECK: @test_relinearize
func.func @test_relinearize(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %key: !cheddar.eval_key,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.relinearize
  %result = cheddar.relinearize %ctx, %ct, %key, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.eval_key, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_relinearize_rescale
func.func @test_relinearize_rescale(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %key: !cheddar.eval_key,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.relinearize_rescale
  %result = cheddar.relinearize_rescale %ctx, %ct, %key, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.eval_key, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// --- Fused compound operations ---

// CHECK: @test_hmult
func.func @test_hmult(
    %ctx: !cheddar.context,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %key: !cheddar.eval_key,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hmult
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key, %out {rescale = true} : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, !cheddar.eval_key, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_hmult_no_rescale
func.func @test_hmult_no_rescale(
    %ctx: !cheddar.context,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %key: !cheddar.eval_key,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key, %out {rescale = false} : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, !cheddar.eval_key, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_hrot_static
func.func @test_hrot_static(
    %ctx: !cheddar.context,
    %ui: !cheddar.user_interface,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hrot
  // CHECK-SAME: static_distance = 5
  %result = cheddar.hrot %ctx, %ui, %ct, %out {static_distance = 5 : i64} : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_hrot_dynamic
func.func @test_hrot_dynamic(
    %ctx: !cheddar.context,
    %ui: !cheddar.user_interface,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>,
    %dist: index) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hrot
  %result = cheddar.hrot %ctx, %ui, %ct, %out, %dist : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, index) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_hrot_add
func.func @test_hrot_add(
    %ctx: !cheddar.context,
    %ui: !cheddar.user_interface,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  %result = cheddar.hrot_add %ctx, %ui, %ct0, %ct1, %out {distance = 3 : i64} : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_hconj
func.func @test_hconj(
    %ctx: !cheddar.context,
    %ui: !cheddar.user_interface,
    %ct: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hconj
  %result = cheddar.hconj %ctx, %ui, %ct, %out : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_hconj_add
func.func @test_hconj_add(
    %ctx: !cheddar.context,
    %ui: !cheddar.user_interface,
    %ct0: tensor<!cheddar.ciphertext>,
    %ct1: tensor<!cheddar.ciphertext>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.hconj_add
  %result = cheddar.hconj_add %ctx, %ui, %ct0, %ct1, %out : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_mad_unsafe
func.func @test_mad_unsafe(
    %ctx: !cheddar.context,
    %acc: tensor<!cheddar.ciphertext>,
    %ct: tensor<!cheddar.ciphertext>,
    %c: tensor<!cheddar.constant>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.mad_unsafe
  %result = cheddar.mad_unsafe %ctx, %acc, %ct, %c : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.constant>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// --- Extension operations ---

// CHECK: @test_boot
func.func @test_boot(
    %ctx: !cheddar.boot_context,
    %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.boot
  %result = cheddar.boot %ctx, %ct, %evk, %out : (!cheddar.boot_context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_linear_transform
func.func @test_linear_transform(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map,
    %diags: tensor<2x4xf64>,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.linear_transform
  // CHECK-SAME: bs = 2
  // CHECK-SAME: diagonal_indices = array<i32: 0, 1>
  // CHECK-SAME: gs = 1
  // CHECK-SAME: level = 5
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diags, %out {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64} : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<2x4xf64>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: @test_eval_poly
func.func @test_eval_poly(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map,
    %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  // CHECK: cheddar.eval_poly
  // CHECK-SAME: coefficients = [1.000000e+00, 2.000000e+00, 3.000000e+00]
  // CHECK-SAME: level = 5
  // CHECK-SAME: outputLevel = 4
  %result = cheddar.eval_poly %ctx, %ct, %evk, %out {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 5 : i64, outputLevel = 4 : i64} : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}
