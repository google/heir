// RUN: heir-opt --cheddar-fuse-ops %s | FileCheck %s

// Test: mult + relinearize + rescale -> hmult(rescale=true)
// CHECK: @test_fuse_hmult_with_rescale
func.func @test_fuse_hmult_with_rescale(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK-NOT: cheddar.rescale
  // CHECK: cheddar.hmult
  // rescale=true is the default and gets elided from the attr-dict
  // CHECK-NOT: rescale = false
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %relin = cheddar.relinearize %ctx, %mult, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  %rescaled = cheddar.rescale %ctx, %relin : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %rescaled : !cheddar.ciphertext
}

// Test: mult + relinearize -> hmult(rescale=false)
// CHECK: @test_fuse_hmult_no_rescale
func.func @test_fuse_hmult_no_rescale(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %relin = cheddar.relinearize %ctx, %mult, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %relin : !cheddar.ciphertext
}

// Test: hrot + add -> hrot_add
// CHECK: @test_fuse_hrot_add
func.func @test_fuse_hrot_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.hrot
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hrot_add
  %rotated = cheddar.hrot %ctx, %ct0, %key {static_shift = 3 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  %sum = cheddar.add %ctx, %rotated, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %sum : !cheddar.ciphertext
}

// Test: hconj + add -> hconj_add
// CHECK: @test_fuse_hconj_add
func.func @test_fuse_hconj_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.hconj
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hconj_add
  %conjugated = cheddar.hconj %ctx, %ct0, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  %sum = cheddar.add %ctx, %conjugated, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %sum : !cheddar.ciphertext
}

// Test: mult + relinearize_rescale -> hmult(rescale=true)
// CHECK: @test_fuse_hmult_with_relin_rescale
func.func @test_fuse_hmult_with_relin_rescale(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize_rescale
  // CHECK: cheddar.hmult
  // CHECK-NOT: rescale = false
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %relin_rescaled = cheddar.relinearize_rescale %ctx, %mult, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %relin_rescaled : !cheddar.ciphertext
}
