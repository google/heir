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

// Encode/encrypt/decode/decrypt: out-param method calls. The attribute
// values for level and scale are inlined into the verbatim format string.

// CHECK: func.func @encode_chain
func.func @encode_chain(%enc: !cheddar.encoder, %msg: tensor<4xf64>,
                        %ui: !cheddar.user_interface)
    -> !cheddar.ciphertext {
  // Note: scale is the C++ `double` value used by CHEDDAR's Encoder
  // (Δ = 2^36 = 68719476736).
  // CHECK: emitc.verbatim "{}->Encode({}, {}, 5, 68719476736.00);"
  %pt = cheddar.encode %enc, %msg {level = 5 : i64, scale = 68719476736.0 : f64}
      : (!cheddar.encoder, tensor<4xf64>) -> !cheddar.plaintext
  // CHECK: emitc.verbatim "{}->Encrypt({}, {});"
  %ct = cheddar.encrypt %ui, %pt
      : (!cheddar.user_interface, !cheddar.plaintext) -> !cheddar.ciphertext
  return %ct : !cheddar.ciphertext
}

// CHECK: func.func @decode_chain
func.func @decode_chain(%enc: !cheddar.encoder, %ui: !cheddar.user_interface,
                        %ct: !cheddar.ciphertext) -> tensor<4xf64> {
  // CHECK: emitc.verbatim "{}->Decrypt({}, {});"
  %pt = cheddar.decrypt %ui, %ct
      : (!cheddar.user_interface, !cheddar.ciphertext) -> !cheddar.plaintext
  // CHECK: emitc.verbatim "{}->Decode({}, {});"
  %msg = cheddar.decode %enc, %pt
      : (!cheddar.encoder, !cheddar.plaintext) -> tensor<4xf64>
  return %msg : tensor<4xf64>
}

// Chaining: each cheddar op produces an `emitc.variable` (the C++ out-param
// destination), the verbatim writes into it, and an `emitc.load` reads it
// back as a value for downstream consumers. The next op's variable is
// declared after the previous load.

// Chaining: each cheddar op declares its own `emitc.variable` (the C++
// out-param destination) and writes into it via verbatim.  Downstream
// consumers reference that variable by name -- the conversion elides the
// `emitc.load` that would otherwise materialise `Ciphertext<word> tmp =
// addv;` (copy-init of a move-only type).
// CHECK: func.func @arith_chain
func.func @arith_chain(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                       %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: %[[ADDV:.*]] = "emitc.variable"
  // CHECK-NEXT: emitc.verbatim "{}->Add({}, {}, {});" args %arg0, %[[ADDV]], %arg1, %arg2
  %r = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  // CHECK-NEXT: %[[MULTV:.*]] = "emitc.variable"
  // CHECK-NEXT: emitc.verbatim "{}->Mult({}, {}, {});" args %arg0, %[[MULTV]], %[[ADDV]], %arg2
  %s = cheddar.mult %ctx, %r, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %s : !cheddar.ciphertext
}

// CHECK: func.func @ct_pt_ct_const
func.func @ct_pt_ct_const(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                          %pt: !cheddar.plaintext, %c: !cheddar.constant)
    -> !cheddar.ciphertext {
  // CHEDDAR's Context overloads Add/Sub/Mult on the second-operand type, so
  // *_plain / *_const ops dispatch to the base name and rely on C++ to pick
  // the right overload.
  // CHECK: emitc.verbatim "{}->Add({}, {}, {});"
  %r1 = cheddar.add_plain %ctx, %ct, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->Sub({}, {}, {});"
  %r2 = cheddar.sub_plain %ctx, %r1, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->Mult({}, {}, {});"
  %r3 = cheddar.mult_plain %ctx, %r2, %pt
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->Add({}, {}, {});"
  %r4 = cheddar.add_const %ctx, %r3, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->Mult({}, {}, {});"
  %r5 = cheddar.mult_const %ctx, %r4, %c
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.constant) -> !cheddar.ciphertext
  return %r5 : !cheddar.ciphertext
}

// CHECK: func.func @unary
func.func @unary(%ctx: !cheddar.context, %ct: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->Neg({}, {});"
  %n = cheddar.neg %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->Rescale({}, {});"
  %r = cheddar.rescale %ctx, %n : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->LevelDown({}, {}, 2);"
  %l = cheddar.level_down %ctx, %r {targetLevel = 2 : i64}
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %l : !cheddar.ciphertext
}

// CHECK: func.func @relin
func.func @relin(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                 %k: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->Relinearize({}, {}, {});"
  %r1 = cheddar.relinearize %ctx, %ct, %k
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  // CHECK: emitc.verbatim "{}->RelinearizeRescale({}, {}, {});"
  %r2 = cheddar.relinearize_rescale %ctx, %r1, %k
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %r2 : !cheddar.ciphertext
}

// HMult: explicit `rescale` flag is rendered as the trailing literal `true`
// or `false`.

// CHECK: func.func @hmult_with_rescale
func.func @hmult_with_rescale(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                              %b: !cheddar.ciphertext, %k: !cheddar.eval_key)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HMult({}, {}, {}, {}, true);"
  %r = cheddar.hmult %ctx, %a, %b, %k {rescale = true}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @hmult_no_rescale
func.func @hmult_no_rescale(%ctx: !cheddar.context, %a: !cheddar.ciphertext,
                            %b: !cheddar.ciphertext, %k: !cheddar.eval_key)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->HMult({}, {}, {}, {}, false);"
  %r = cheddar.hmult %ctx, %a, %b, %k {rescale = false}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// HRot static-distance: the distance is baked into the format string twice,
// once for the key lookup and once for the rotation argument.

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
// SSA result aliases the accumulator input. Subsequent uses of `%r` resolve
// to the same C++ variable as `%acc`.

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
func.func @boot(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim "{}->Boot({}, {}, {});"
  %r = cheddar.boot %ctx, %ct, %evk
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// Array attrs are inlined into the verbatim format string with doubled
// braces so that `emitc.verbatim` interprets them as literal `{` / `}` and
// renders the initializer list correctly. We check the substring rather
// than the full format string (FileCheck's `{{...}}` is a regex marker).

// CHECK: func.func @linear_transform
func.func @linear_transform(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                            %evk: !cheddar.evk_map, %d: tensor<2x4xf64>)
    -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim
  // CHECK-SAME: ->LinearTransform
  // CHECK-SAME: 0, 1
  // CHECK-SAME: 5, 0
  %r = cheddar.linear_transform %ctx, %ct, %evk, %d
      {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, logBabyStepGiantStepRatio = 0 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<2x4xf64>)
      -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// CHECK: func.func @eval_poly
func.func @eval_poly(%ctx: !cheddar.context, %ct: !cheddar.ciphertext,
                     %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // CHECK: emitc.verbatim
  // CHECK-SAME: ->EvalPoly
  // CHECK-SAME: 1.0, 2.0, 3.0
  // CHECK-SAME: 4
  %r = cheddar.eval_poly %ctx, %ct, %evk
      {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 4 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
