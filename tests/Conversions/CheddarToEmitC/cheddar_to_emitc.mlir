// RUN: heir-opt --cheddar-to-emitc %s | FileCheck %s

// CreateContext is a static factory. PrepareRotKey is a void method. The
// getter-style setup ops (create_user_interface / get_encoder / get_evk_map /
// get_mult_key) are unsupported and rejected; see unsupported_getters.mlir.

// CHECK: func.func @create_context
// CHECK: emitc.call_opaque "Context<word>::Create"
func.func @create_context(%params: !cheddar.parameter) -> !cheddar.context {
  %ctx = cheddar.create_context %params : (!cheddar.parameter) -> !cheddar.context
  return %ctx : !cheddar.context
}

// CHECK: func.func @prepare_keys
func.func @prepare_keys(%ui: !cheddar.user_interface) {
  // CHECK: emitc.verbatim "{}->PrepareRotationKey(3, 5);" args %arg0
  cheddar.prepare_rot_key %ui {distance = 3 : i64, maxLevel = 5 : i64} : (!cheddar.user_interface) -> ()
  return
}

// Encrypt/decrypt are out-param method calls (emitc.member_call_opaque, which
// picks `.`/`->` from the receiver type). Encode/decode bridge a message
// buffer through a std::vector<Complex> and so stay as emitc.verbatim; the
// emitter emits encode's `scale` attribute verbatim (here Δ = 2^36 = 68719476736).

// CHECK: func.func @encode_chain
func.func @encode_chain(%enc: !cheddar.encoder, %msg: tensor<4xf64>,
                        %ui: !cheddar.user_interface)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}.Encode({}, 5, 68719476736, {});"
  %pt = cheddar.encode %enc, %msg {level = 5 : i64, scale = 68719476736.0 : f64}
      : (!cheddar.encoder, tensor<4xf64>) -> !cheddar.plaintext
  // CHECK: emitc.member_call_opaque %arg2 "Encrypt"
  %ct = cheddar.encrypt %ui, %pt
      : (!cheddar.user_interface, !cheddar.plaintext) -> !cheddar.ciphertext
  return %ct : !cheddar.ciphertext
}

// Decode is DPS: %dst is the destination buffer (a <1x...xN> memref -> emitc
// array), the result aliases it. The emitter fills a temporary
// std::vector<Complex>, then copies the real parts into %dst.

// CHECK: func.func @decode_chain
func.func @decode_chain(%enc: !cheddar.encoder, %ui: !cheddar.user_interface,
                        %ct: !cheddar.ciphertext, %dst: memref<1x4xf32>) {
  // CHECK: emitc.member_call_opaque %arg1 "Decrypt"
  %pt = cheddar.decrypt %ui, %ct
      : (!cheddar.user_interface, !cheddar.ciphertext) -> !cheddar.plaintext
  // CHECK: emitc.verbatim "{}.Decode({}, {});"
  %msg = cheddar.decode %enc, %pt, %dst
      : (!cheddar.encoder, !cheddar.plaintext, memref<1x4xf32>) -> memref<1x4xf32>
  return
}

// Chaining: each out-param op declares its own `emitc.variable` (the C++
// destination) and the member_call_opaque writes into it. Downstream consumers
// reference that variable by name -- the conversion elides the load that would
// otherwise copy-init a move-only type.
// CHECK: func.func @arith_chain
func.func @arith_chain(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                       %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: %[[ADDV:.*]] = "emitc.variable"
  // CHECK: emitc.member_call_opaque %arg0 "Add"(%[[ADDV]], %arg1, %arg2)
  %r = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  // CHECK: %[[MULTV:.*]] = "emitc.variable"
  // CHECK: emitc.member_call_opaque %arg0 "Mult"(%[[MULTV]], %[[ADDV]], %arg2)
  %s = cheddar.mult %ctx, %r, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %s : !cheddar.ciphertext
}

// CHECK: func.func @ct_pt_ct_const
func.func @ct_pt_ct_const(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                          %pt: !cheddar.plaintext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  // CHEDDAR's Context overloads Add/Sub/Mult on the second-operand type, so
  // *_plain / *_const ops dispatch to the base method name and rely on C++ to
  // pick the right overload.
  // CHECK: emitc.member_call_opaque %arg0 "Add"
  %r1 = cheddar.add_plain %ctx, %ct, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  // CHECK: emitc.member_call_opaque %arg0 "Sub"
  %r2 = cheddar.sub_plain %ctx, %r1, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  // CHECK: emitc.member_call_opaque %arg0 "Mult"
  %r3 = cheddar.mult_plain %ctx, %r2, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  // CHECK: emitc.member_call_opaque %arg0 "Add"
  %r4 = cheddar.add_const %ctx, %r3, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  // CHECK: emitc.member_call_opaque %arg0 "Mult"
  %r5 = cheddar.mult_const %ctx, %r4, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %r5 : !cheddar.ciphertext
}

// CHECK: func.func @unary
func.func @unary(%ctx: !cheddar.context, %ct: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  // CHECK: emitc.member_call_opaque %arg0 "Neg"
  %n = cheddar.neg %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  // CHECK: emitc.member_call_opaque %arg0 "Rescale"
  %r = cheddar.rescale %ctx, %n : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  // The target level is appended as a trailing opaque constant argument.
  // CHECK: emitc.member_call_opaque %arg0 "LevelDown"
  // CHECK-SAME: #emitc.opaque<"2">
  %l = cheddar.level_down %ctx, %r {targetLevel = 2 : i64}
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %l : !cheddar.ciphertext
}

// CHECK: func.func @relin
func.func @relin(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                 %k: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: emitc.member_call_opaque %arg0 "Relinearize"
  %r1 = cheddar.relinearize %ctx, %ct, %k
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  // CHECK: emitc.member_call_opaque %arg0 "RelinearizeRescale"
  %r2 = cheddar.relinearize_rescale %ctx, %r1, %k
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %r2 : !cheddar.ciphertext
}

// HMult: the explicit `rescale` flag is appended as the trailing opaque
// constant argument `true` / `false`.

// CHECK: func.func @hmult_with_rescale
func.func @hmult_with_rescale(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                              %b: !cheddar.ciphertext, %k: !cheddar.eval_key)
    -> !cheddar.ciphertext {
  // CHECK: emitc.member_call_opaque %arg0 "HMult"
  // CHECK-SAME: #emitc.opaque<"true">
  %r = cheddar.hmult %ctx, %a, %b, %k {rescale = true}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @hmult_no_rescale
func.func @hmult_no_rescale(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                            %b: !cheddar.ciphertext, %k: !cheddar.eval_key)
    -> !cheddar.ciphertext {
  // CHECK: emitc.member_call_opaque %arg0 "HMult"
  // CHECK-SAME: #emitc.opaque<"false">
  %r = cheddar.hmult %ctx, %a, %b, %k {rescale = false}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// HRot/HConj keep their `emitc.verbatim` form: the rotation/conjugation key is
// a nested `{}->GetRotationKey(d)` lookup that a plain member_call_opaque arg
// list can't express. Static-distance bakes the distance into the format
// string twice (key lookup + rotation arg).

// CHECK: func.func @hrot_static
func.func @hrot_static(%ctx: !cheddar.context, %ui: !cheddar.user_interface,
                       %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HRot({}, {}, {}->GetRotationKey(5), 5);"
  %r = cheddar.hrot %ctx, %ct {static_distance = 5 : i64}
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// HRot dynamic-distance: the SSA distance value appears twice in the
// `args` list (once for the key, once for the rotation argument).

// CHECK: func.func @hrot_dynamic
func.func @hrot_dynamic(%ctx: !cheddar.context, %ui: !cheddar.user_interface,
                        %ct: !cheddar.ciphertext, %d: index)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HRot({}, {}, {}->GetRotationKey({}), {});"
  // CHECK-SAME: %arg3, %arg3
  %r = cheddar.hrot %ctx, %ct, %d
      : (!cheddar.context, !cheddar.ciphertext, index) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @hrot_add
func.func @hrot_add(%ctx: !cheddar.context, %ui: !cheddar.user_interface,
                    %a: !cheddar.ciphertext, %b: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HRotAdd({}, {}, {}, {}->GetRotationKey(7), 7);"
  %r = cheddar.hrot_add %ctx, %a, %b {distance = 7 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @hconj
func.func @hconj(%ctx: !cheddar.context, %ui: !cheddar.user_interface,
                 %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HConj({}, {}, {}->GetConjugationKey());"
  %r = cheddar.hconj %ctx, %ct
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @hconj_add
func.func @hconj_add(%ctx: !cheddar.context, %ui: !cheddar.user_interface,
                     %a: !cheddar.ciphertext, %b: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}->GetConjugationKey());"
  %r = cheddar.hconj_add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// MadUnsafe is in-place: no new variable is declared for the result; the
// SSA result aliases the accumulator input. It stays an emitc.verbatim.

// CHECK: func.func @mad
func.func @mad(%ctx: !cheddar.context, %acc: !cheddar.ciphertext,
               %in: !cheddar.ciphertext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->MadUnsafe({}, {}, {});" args %arg0, %arg1, %arg2, %arg3
  // CHECK-NOT: emitc.variable
  %r = cheddar.mad_unsafe %ctx, %acc, %in, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.constant)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @boot
func.func @boot(%ctx: !cheddar.boot_context, %ct: !cheddar.ciphertext,
                %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // Boot is a BootContext method, and cheddar.boot requires a
  // !cheddar.boot_context (lowered to BootContext<word>*), so the call needs no
  // downcast. See ConvertBoot in CheddarToEmitC.cpp.
  // CHECK: emitc.verbatim "{}->Boot({}, {}, {});"
  %r = cheddar.boot %ctx, %ct, %evk
      : (!cheddar.boot_context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// Array/list attrs are appended as one trailing opaque constant argument (an
// initializer-list literal). We check the substrings rather than the full
// opaque (FileCheck's `{{...}}` is a regex marker, and the literal contains
// `{{`).

// CHECK: func.func @linear_transform
func.func @linear_transform(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                            %evk: !cheddar.evk_map, %d: tensor<2x4xf64>)
    -> !cheddar.ciphertext {
  // There is no Context::LinearTransform method (it's a class), so the emitter
  // lowers to one structured call to the HEIR-side RunLinearTransform shim,
  // carrying {indices}, level, bs, gs as a trailing literal and the slot width +
  // `word` as template args.
  // CHECK: emitc.call_opaque "RunLinearTransform"
  // CHECK-SAME: 0, 1}, 5, 2, 1
  // CHECK-SAME: word
  %r = cheddar.linear_transform %ctx, %ct, %evk, %d
      {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<2x4xf64>)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @eval_poly
func.func @eval_poly(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                     %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // No Context::EvalPoly method (it's a class), so the emitter lowers to one
  // structured call to the HEIR-side RunEvalPoly shim, carrying {coefficients},
  // level and outputLevel as a trailing literal and `word` as a template arg.
  // CHECK: emitc.call_opaque "RunEvalPoly"
  // CHECK-SAME: 2, 3}, 4, 3
  // CHECK-SAME: word
  %r = cheddar.eval_poly %ctx, %ct, %evk
      {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 4 : i64, outputLevel = 3 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
