// RUN: heir-opt --cheddar-to-emitc %s | FileCheck %s

// The full pipeline on input with no arith, scf or math ops. The boundary pass
// does not load those dialects, and --convert-to-emitc fails on filter
// dialects that are not loaded, so this catches a pipeline change that stops
// an earlier pass from loading them.

// CHECK: func.func @neg(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>, %[[IN:.*]]: !emitc.opaque<"const Ciphertext<word>&">, %[[OUT:.*]]: !emitc.opaque<"Ciphertext<word>&"> {bufferize.result})
// CHECK: emitc.member_call_opaque %[[CTX]] "Neg"(%[[OUT]], %[[IN]])
func.func @neg(%ctx: !cheddar.context, %in: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %d = tensor.empty() : tensor<!cheddar.ciphertext>
  %r = cheddar.neg %ctx, %in, %d : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %r : tensor<!cheddar.ciphertext>
}
