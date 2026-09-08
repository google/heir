// RUN: heir-opt --cheddar-bufferize --split-input-file --verify-diagnostics %s

// A user interface is move-only: a read-write destination that is not writable
// would need a copy, so it must bufferize in place.
func.func @borrowed_ui(%ui: tensor<!cheddar.user_interface> {bufferization.writable = false}) -> tensor<!cheddar.user_interface> {
  // expected-error@+1 {{move-only read-write destination must bufferize in-place}}
  %updated = cheddar.prepare_rot_key %ui {distance = 7 : i64, maxLevel = 13 : i64} : (tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %updated : tensor<!cheddar.user_interface>
}

// -----

// A loop must yield the buffer of its iter_arg, not a fresh allocation.
func.func @fresh_loop_result(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>, %upper: index) -> tensor<!cheddar.ciphertext> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%iter = %input) -> tensor<!cheddar.ciphertext> {
    %empty = tensor.empty() : tensor<!cheddar.ciphertext>
    %next = cheddar.neg %ctx, %iter, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    // expected-error@+1 {{Yield operand #0 is not equivalent to the corresponding iter bbArg}}
    scf.yield %next : tensor<!cheddar.ciphertext>
  }
  return %result : tensor<!cheddar.ciphertext>
}
