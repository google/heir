// Input for the cheddar-to-emitc *compile* test (see BUILD): every function
// here is lowered to C++ and compiled against cheddar_stub.h. The point is to
// exercise the emitter's move/const handling on the op surface that real
// kernels use, with ctx / user_interface / keys / evk_map taken as function
// arguments (the shape a CKKS-to-Cheddar lowering produces).
//
// The setup/getter ops (create_context, create_user_interface, get_encoder,
// get_evk_map, get_mult_key, encode/decode) are intentionally absent: that
// part of the emitter has independent, pre-existing API mismatches and is
// tracked separately.

// Add / Sub / Mult chained on ciphertexts.
func.func @arith(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                 %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %0 = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %1 = cheddar.sub %ctx, %0, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %2 = cheddar.mult %ctx, %1, %a
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %2 : !cheddar.ciphertext
}

// ct+pt and ct+const overloaded dispatch.
func.func @ct_pt_const(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                       %pt: !cheddar.plaintext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  %0 = cheddar.add_plain %ctx, %ct, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  %1 = cheddar.sub_plain %ctx, %0, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  %2 = cheddar.mult_plain %ctx, %1, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  %3 = cheddar.add_const %ctx, %2, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  %4 = cheddar.mult_const %ctx, %3, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %4 : !cheddar.ciphertext
}

// Unary ops.
func.func @unary(%ctx: !cheddar.context, %ct: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  %0 = cheddar.neg %ctx, %ct
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  %1 = cheddar.rescale %ctx, %0
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  %2 = cheddar.level_down %ctx, %1 {targetLevel = 2 : i64}
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %2 : !cheddar.ciphertext
}

// Relinearize / RelinearizeRescale with an evaluation-key argument.
func.func @relin(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                 %k: !cheddar.eval_key) -> !cheddar.ciphertext {
  %0 = cheddar.relinearize %ctx, %ct, %k
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  %1 = cheddar.relinearize_rescale %ctx, %0, %k
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %1 : !cheddar.ciphertext
}

// HMult with an evaluation-key argument.
func.func @hmult(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                 %b: !cheddar.ciphertext, %k: !cheddar.eval_key)
    -> !cheddar.ciphertext {
  %0 = cheddar.hmult %ctx, %a, %b, %k {rescale = true}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key)
      -> !cheddar.ciphertext
  return %0 : !cheddar.ciphertext
}

// Rotation / conjugation: the key is looked up inline via the UserInterface
// argument, so these functions must carry a user_interface arg.
func.func @rotations(%ctx: !cheddar.context, %ui: !cheddar.user_interface,
                     %a: !cheddar.ciphertext, %b: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  %0 = cheddar.hrot %ctx, %a {static_distance = 5 : i64}
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  %1 = cheddar.hrot_add %ctx, %0, %b {distance = 7 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %2 = cheddar.hconj %ctx, %1
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  %3 = cheddar.hconj_add %ctx, %2, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %3 : !cheddar.ciphertext
}

// mad_unsafe with a *local* accumulator (the result of add): the accumulator
// is a plain local lvalue, so MadUnsafe(acc, ...) binds fine. This path
// already compiled; it's here as the control case.
func.func @mad_local(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                     %b: !cheddar.ciphertext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  %acc = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %r = cheddar.mad_unsafe %ctx, %acc, %a, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.constant)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// mad_unsafe with the accumulator coming straight from a *function argument*
// (finding 1). The accumulator is mutated in place by MadUnsafe and then
// returned, so it must be lifted to a mutable `Ct&` -- not the `const Ct&`
// that the by-value-arg tightening would otherwise produce.
func.func @mad_arg(%ctx: !cheddar.context, %acc: !cheddar.ciphertext,
                   %in: !cheddar.ciphertext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  %r = cheddar.mad_unsafe %ctx, %acc, %in, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.constant)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// Identity: returning a move-only argument unchanged must lift the arg to an
// in-place `Ct&` out-param, not copy.
func.func @identity(%ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  return %ct : !cheddar.ciphertext
}

// Bootstrapping-family ops taking an EvkMap argument (const EvkMap& at the C++
// boundary).
func.func @boot(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  %0 = cheddar.boot %ctx, %ct, %evk
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %0 : !cheddar.ciphertext
}

// The diagonals are a memref (the bufferized form the real pipeline produces);
// the emitter renders it as a 2D `double[][W]` array passed to RunLinearTransform.
func.func @linear_transform(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                            %evk: !cheddar.evk_map, %d: memref<2x4xf64>)
    -> !cheddar.ciphertext {
  %0 = cheddar.linear_transform %ctx, %ct, %evk, %d
      {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, memref<2x4xf64>)
      -> !cheddar.ciphertext
  return %0 : !cheddar.ciphertext
}

func.func @eval_poly(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                     %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  %0 = cheddar.eval_poly %ctx, %ct, %evk
      {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 4 : i64, outputLevel = 3 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %0 : !cheddar.ciphertext
}

// Encrypt / Decrypt out-param calls on the UserInterface.
func.func @encrypt_decrypt(%ui: !cheddar.user_interface, %pt: !cheddar.plaintext,
                           %ct: !cheddar.ciphertext)
    -> (!cheddar.ciphertext, !cheddar.plaintext) {
  %0 = cheddar.encrypt %ui, %pt
      : (!cheddar.user_interface, !cheddar.plaintext) -> !cheddar.ciphertext
  %1 = cheddar.decrypt %ui, %ct
      : (!cheddar.user_interface, !cheddar.ciphertext) -> !cheddar.plaintext
  return %0, %1 : !cheddar.ciphertext, !cheddar.plaintext
}

// Bufferized loop kernel: `scf.for` + `arith` index + `memref.store` of a
// move-only payload into an output buffer. The whole loop lowers to EmitC in
// the single `cheddar-to-emitc` conversion (SCF/Arith patterns run alongside
// the cheddar patterns). The output memref is *written*, so its boundary type
// must be a mutable `std::array<Ciphertext<word>, 8>&` (a `const` one would
// reject the `arr[i] = std::move(...)` store -- the const-correctness guard).
func.func @loop_store(%ctx: !cheddar.context, %in: !cheddar.ciphertext,
                      %out: memref<8x!cheddar.ciphertext>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %v = cheddar.add %ctx, %in, %in
        : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
    memref.store %v, %out[%i] : memref<8x!cheddar.ciphertext>
  }
  return
}
