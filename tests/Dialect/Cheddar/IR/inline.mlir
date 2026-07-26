// RUN: heir-opt %s --inline | FileCheck %s

// The cheddar inliner interface inlines pure ciphertext-algebra ops but refuses
// to *duplicate* stateful setup/keygen ops.

// A single-use callee of pure ops is inlined: its body is moved into the caller
// and the call disappears.
// CHECK: func.func @use_pure
// CHECK: cheddar.add
// CHECK-NOT: call @pure_callee
func.func @pure_callee(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                       %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %r = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
func.func @use_pure(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                    %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %r = func.call @pure_callee(%ctx, %a, %b)
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// A keygen callee invoked from two sites would have to be *cloned* to inline.
// Cloning prepare_rot_key (a key-generation side effect) is illegal, so both
// calls are left in place and the op is never duplicated into the caller.
// CHECK: func.func @use_keygen_twice
// CHECK: call @keygen_callee
// CHECK: call @keygen_callee
func.func @keygen_callee(%ui: !cheddar.user_interface) {
  cheddar.prepare_rot_key %ui {distance = 1 : i64, maxLevel = 5 : i64}
      : (!cheddar.user_interface) -> ()
  return
}
func.func @use_keygen_twice(%ui: !cheddar.user_interface) {
  func.call @keygen_callee(%ui) : (!cheddar.user_interface) -> ()
  func.call @keygen_callee(%ui) : (!cheddar.user_interface) -> ()
  return
}
