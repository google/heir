// RUN: heir-translate --emit-cheddar %s | FileCheck %s

// CHECK: #include "core/Context.h"
// CHECK: using namespace cheddar;
// CHECK: using word = uint64_t;

// --- Binary ct-ct operations ---

// CHECK: Ct test_add(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]])
func.func @test_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
  // CHECK: double lhs_scale = [[CT0]].GetScale();
  // CHECK-NEXT: double rhs_scale = [[CT1]].GetScale();
  // CHECK: [[CTX]]->Add([[RES]], [[CT0]], [[CT1]]);
  %result = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_sub(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]])
func.func @test_sub(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
  // CHECK: double lhs_scale = [[CT0]].GetScale();
  // CHECK-NEXT: double rhs_scale = [[CT1]].GetScale();
  // CHECK: [[CTX]]->Sub([[RES]], [[CT0]], [[CT1]]);
  %result = cheddar.sub %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_mult(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]])
func.func @test_mult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Mult([[RES]], [[CT0]], [[CT1]]);
  %result = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: int32_t test_floor_div_mul(
// CHECK-SAME: int32_t [[A:.*]], int32_t [[B:.*]])
func.func @test_floor_div_mul(%a: i32, %b: i32) -> i32 {
  // CHECK: int32_t [[Q:.*]] = ([[A]] / [[B]]) - (([[A]] % [[B]] != 0) && (([[A]] < 0) != ([[B]] < 0)));
  %q = arith.floordivsi %a, %b : i32
  // CHECK: int32_t [[M:.*]] = [[Q]] * [[B]];
  %m = arith.muli %q, %b : i32
  return %m : i32
}

// --- Ct-pt / ct-const operations ---

// CHECK: Ct test_add_plain(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], Pt& [[PT:.*]])
func.func @test_add_plain(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[PT]].SetScale([[CT]].GetScale());
  // CHECK-NEXT: [[CTX]]->Add([[RES]], [[CT]], [[PT]]);
  %result = cheddar.add_plain %ctx, %ct, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_sub_plain(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], Pt& [[PT:.*]])
func.func @test_sub_plain(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
  // CHECK: double lhs_scale = [[CT]].GetScale();
  // CHECK-NEXT: double rhs_scale = [[PT]].GetScale();
  // CHECK: [[CTX]]->Sub([[RES]], [[CT]], [[PT]]);
  %result = cheddar.sub_plain %ctx, %ct, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_mult_plain(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], Pt& [[PT:.*]])
func.func @test_mult_plain(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Mult([[RES]], [[CT]], [[PT]]);
  %result = cheddar.mult_plain %ctx, %ct, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_add_const(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], const Const& [[C:.*]])
func.func @test_add_const(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %c: !cheddar.constant) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Add([[RES]], [[CT]], [[C]]);
  %result = cheddar.add_const %ctx, %ct, %c : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_mult_const(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], const Const& [[C:.*]])
func.func @test_mult_const(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %c: !cheddar.constant) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Mult([[RES]], [[CT]], [[C]]);
  %result = cheddar.mult_const %ctx, %ct, %c : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Unary operations ---

// CHECK: Ct test_neg(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]])
func.func @test_neg(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Neg([[RES]], [[CT]]);
  %result = cheddar.neg %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_rescale(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]])
func.func @test_rescale(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Rescale([[RES]], [[CT]]);
  %result = cheddar.rescale %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_level_down(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]])
func.func @test_level_down(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->LevelDown([[RES]], [[CT]], 3);
  %result = cheddar.level_down %ctx, %ct {targetLevel = 3 : i64} : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Key-switching operations ---

// CHECK: Ct test_relinearize(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], const Evk& [[KEY:.*]])
func.func @test_relinearize(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Relinearize([[RES]], [[CT]], [[KEY]]);
  %result = cheddar.relinearize %ctx, %ct, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_relinearize_rescale(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], const Evk& [[KEY:.*]])
func.func @test_relinearize_rescale(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->RelinearizeRescale([[RES]], [[CT]], [[KEY]]);
  %result = cheddar.relinearize_rescale %ctx, %ct, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Fused compound operations ---

// CHECK: Ct test_hmult(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]], const Evk& [[KEY:.*]])
func.func @test_hmult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->HMult([[RES]], [[CT0]], [[CT1]], [[KEY]], true);
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = true} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hmult_no_rescale(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]], const Evk& [[KEY:.*]])
func.func @test_hmult_no_rescale(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->HMult([[RES]], [[CT0]], [[CT1]], [[KEY]], false);
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = false} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hrot(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], const Evk& [[KEY:.*]])
func.func @test_hrot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->HRot([[RES]], [[CT]], [[KEY]], 5);
  %result = cheddar.hrot %ctx, %ct, %key {static_shift = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hrot_add(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]], const Evk& [[KEY:.*]])
func.func @test_hrot_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->HRotAdd([[RES]], [[CT0]], [[CT1]], [[KEY]], 3);
  %result = cheddar.hrot_add %ctx, %ct0, %ct1, %key {distance = 3 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hconj(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT:.*]], const Evk& [[KEY:.*]])
func.func @test_hconj(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->HConj([[RES]], [[CT]], [[KEY]]);
  %result = cheddar.hconj %ctx, %ct, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hconj_add(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[CT0:.*]], const Ct& [[CT1:.*]], const Evk& [[KEY:.*]])
func.func @test_hconj_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->HConjAdd([[RES]], [[CT0]], [[CT1]], [[KEY]]);
  %result = cheddar.hconj_add %ctx, %ct0, %ct1, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_mad_unsafe(
// CHECK-SAME: CtxPtr [[CTX:.*]], const Ct& [[ACC:.*]], const Ct& [[CT:.*]], const Const& [[C:.*]])
func.func @test_mad_unsafe(
    %ctx: !cheddar.context,
    %acc: !cheddar.ciphertext,
    %ct: !cheddar.ciphertext,
    %c: !cheddar.constant) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[CTX]]->Copy([[RES]], [[ACC]]);
  // CHECK-NEXT: [[CTX]]->MadUnsafe([[RES]], [[CT]], [[C]]);
  %result = cheddar.mad_unsafe %ctx, %acc, %ct, %c : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// --- Setup operations ---

// CHECK: void test_setup(
// CHECK-SAME: const Param& [[PARAMS:.*]])
func.func @test_setup(%params: !cheddar.parameter) {
  // CHECK: auto [[CTX:.*]] = Context<word>::Create([[PARAMS]]);
  %ctx = cheddar.create_context %params : (!cheddar.parameter) -> !cheddar.context
  // CHECK: UI [[UI:.*]]([[CTX]]);
  %ui = cheddar.create_user_interface %ctx : (!cheddar.context) -> !cheddar.user_interface
  // CHECK: auto& [[ENC:.*]] = [[CTX]]->encoder_;
  %enc = cheddar.get_encoder %ctx : (!cheddar.context) -> !cheddar.encoder
  // CHECK: const auto& [[EVK:.*]] = [[UI]].GetEvkMap();
  %evk = cheddar.get_evk_map %ui : (!cheddar.user_interface) -> !cheddar.evk_map
  // CHECK: const auto& [[MKEY:.*]] = [[UI]].GetMultiplicationKey();
  %mkey = cheddar.get_mult_key %ui : (!cheddar.user_interface) -> !cheddar.eval_key
  // CHECK: const auto& [[RKEY:.*]] = [[UI]].GetRotationKey(7);
  %rkey = cheddar.get_rot_key %ui {distance = 7 : i64} : (!cheddar.user_interface) -> !cheddar.eval_key
  // CHECK: const auto& [[CKEY:.*]] = [[UI]].GetConjugationKey();
  %ckey = cheddar.get_conj_key %ui : (!cheddar.user_interface) -> !cheddar.eval_key
  // CHECK: [[UI]].PrepareRotationKey(3, 10);
  cheddar.prepare_rot_key %ui {distance = 3 : i64, maxLevel = 10 : i64} : (!cheddar.user_interface) -> ()
  return
}

// --- Encrypt / Decrypt ---

// CHECK: Ct test_encrypt(
// CHECK-SAME: UI& [[UI:.*]], Pt& [[PT:.*]])
func.func @test_encrypt(
    %ui: !cheddar.user_interface,
    %pt: !cheddar.plaintext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: [[UI]].Encrypt([[RES]], [[PT]]);
  %ct = cheddar.encrypt %ui, %pt : (!cheddar.user_interface, !cheddar.plaintext) -> !cheddar.ciphertext
  return %ct : !cheddar.ciphertext
}

// CHECK: Pt test_decrypt(
// CHECK-SAME: UI& [[UI:.*]], const Ct& [[CT:.*]])
func.func @test_decrypt(
    %ui: !cheddar.user_interface,
    %ct: !cheddar.ciphertext) -> !cheddar.plaintext {
  // CHECK: Pt [[RES:.*]];
  // CHECK-NEXT: [[UI]].Decrypt([[RES]], [[CT]]);
  %pt = cheddar.decrypt %ui, %ct : (!cheddar.user_interface, !cheddar.ciphertext) -> !cheddar.plaintext
  return %pt : !cheddar.plaintext
}
