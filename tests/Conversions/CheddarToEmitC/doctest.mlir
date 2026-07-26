// RUN: heir-opt --cheddar-to-emitc %s | FileCheck %s

// A server-side op (here cheddar.add) lowers to an out-parameter method call on
// the Context: the SSA result becomes a trailing `Ciphertext<word>&` argument,
// the `Context::Add` call writes into a local, and that local is moved into the
// out-parameter (CHEDDAR's Ciphertext is move-only).

// CHECK:      func.func @add(
// CHECK-SAME:   !emitc.ptr<!emitc.opaque<"Context<word>">>
// CHECK-SAME:   !emitc.opaque<"Ciphertext<word>&">
// CHECK:        emitc.member_call_opaque %arg0 "Add"
// CHECK:        emitc.verbatim "{} = std::move({});"
func.func @add(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
               %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %c = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %c : !cheddar.ciphertext
}
