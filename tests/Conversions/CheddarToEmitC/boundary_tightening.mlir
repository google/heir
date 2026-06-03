// RUN: heir-opt --cheddar-to-emitc --split-input-file %s | FileCheck %s

// The mad_unsafe accumulator comes from a function argument; it is mutated in
// place by MadUnsafe and returned, so it must be lifted to a mutable
// `Ciphertext<word>&` (the `<"Ciphertext` prefix anchors the match away from
// the `const Ciphertext` inputs) and the result dropped -- no out-param, no
// `std::move` at the return.
// CHECK: func.func @mad_arg
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"const Constant<word>&">
// CHECK: emitc.verbatim "{}->MadUnsafe({}, {}, {});"
// CHECK-NOT: std::move
func.func @mad_arg(%ctx: !cheddar.context, %acc: !cheddar.ciphertext,
                   %in: !cheddar.ciphertext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  %r = cheddar.mad_unsafe %ctx, %acc, %in, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.constant)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// -----

// Returning a move-only argument unchanged lifts it to an in-place
// `Ciphertext<word>&` out-param with no result and no copy.
// CHECK: func.func @identity
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK-NOT: std::move
func.func @identity(%ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  return %ct : !cheddar.ciphertext
}

// -----

// An EvkMap argument is move-only, so it tightens to `const EvkMap<word>&`
// rather than staying a by-value parameter.
// CHECK: func.func @boot
// CHECK-SAME: !emitc.opaque<"const EvkMap<word>&">
func.func @boot(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  %0 = cheddar.boot %ctx, %ct, %evk
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %0 : !cheddar.ciphertext
}

// -----

// An Encoder argument (a non-assignable view) tightens to `const Encoder<word>&`.
// CHECK: func.func @encoder_arg
// CHECK-SAME: !emitc.opaque<"const Encoder<word>&">
func.func @encoder_arg(%enc: !cheddar.encoder, %ct: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  return %ct : !cheddar.ciphertext
}
