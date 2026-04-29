// RUN: heir-translate --emit-cheddar %s | FileCheck %s

// Test scf.for emission
// CHECK: Ct test_for_loop
func.func @test_for_loop(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // CHECK: for (int64_t {{.*}} = 0; {{.*}} < 4; {{.*}} += 1) {
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %ct0) -> !cheddar.ciphertext {
    // CHECK: {{.*}}->Add(
    %sum = cheddar.add %ctx, %acc, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
    scf.yield %sum : !cheddar.ciphertext
  }
  // CHECK: }
  return %result : !cheddar.ciphertext
}

// Test scf.if emission
// CHECK: Ct test_if
func.func @test_if(
    %ctx: !cheddar.context,
    %cond: i1,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK: if ({{.*}}) {
  %result = scf.if %cond -> !cheddar.ciphertext {
    // CHECK: {{.*}}->Add(
    %sum = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
    scf.yield %sum : !cheddar.ciphertext
  } else {
    // CHECK: } else {
    // CHECK: {{.*}}->Sub(
    %diff = cheddar.sub %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
    scf.yield %diff : !cheddar.ciphertext
  }
  // CHECK: }
  return %result : !cheddar.ciphertext
}

// Test that numeric tensor updates carried through SCF loops reuse the moved
// vector instead of deep-copying it on every tensor.insert.
// CHECK: std::vector<double> test_tensor_insert_move
func.func @test_tensor_insert_move() -> tensor<4xf32> {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %c1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1_idx = arith.constant 1 : index
  %result = scf.for %i = %c0 to %c4 step %c1_idx iter_args(%acc = %cst) -> tensor<4xf32> {
    // CHECK: auto [[INS:.*]] = std::move([[ACC:.*]]);
    // CHECK-NEXT: [[INS]][{{.*}}] = {{.*}};
    %inserted = tensor.insert %c1 into %acc[%i] : tensor<4xf32>
    scf.yield %inserted : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}

// Test that numeric tensor.insert_slice also reuses the moved vector when the
// destination is a loop-carried local.
// CHECK: std::vector<double> test_tensor_insert_slice_move
func.func @test_tensor_insert_slice_move() -> tensor<4xf32> {
  %dest = arith.constant dense<0.0> : tensor<4xf32>
  %src = arith.constant dense<1.0> : tensor<2xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %dest) -> tensor<4xf32> {
    // CHECK: auto [[INS_SLICE:.*]] = std::move([[ACC_SLICE:.*]]);
    // CHECK-NEXT: std::copy({{.*}}.begin(), {{.*}}.end(), [[INS_SLICE]].begin() + 1);
    %inserted = tensor.insert_slice %src into %acc[1] [2] [1] : tensor<2xf32> into tensor<4xf32>
    scf.yield %inserted : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}
