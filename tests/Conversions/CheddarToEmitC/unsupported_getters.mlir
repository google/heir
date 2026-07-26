// RUN: heir-opt --cheddar-to-emitc --split-input-file --verify-diagnostics %s

// The getter-style setup ops return a const reference to a move-only /
// non-assignable CHEDDAR value (EvkMap, EvaluationKey, Encoder, UserInterface),
// which can't be materialised into a local without a copy. The lowering
// rejects them rather than emit uncompilable C++. Real kernels pass these as
// function arguments or look them up inline (like HRot's rotation-key lookup).

func.func @get_evk_map(%ui: !cheddar.user_interface) -> !cheddar.evk_map {
  // expected-error @below {{lowering of 'cheddar.get_evk_map' is not supported}}
  %m = cheddar.get_evk_map %ui : (!cheddar.user_interface) -> !cheddar.evk_map
  return %m : !cheddar.evk_map
}

// -----

func.func @get_mult_key(%ui: !cheddar.user_interface) -> !cheddar.eval_key {
  // expected-error @below {{lowering of 'cheddar.get_mult_key' is not supported}}
  %k = cheddar.get_mult_key %ui : (!cheddar.user_interface) -> !cheddar.eval_key
  return %k : !cheddar.eval_key
}

// -----

func.func @get_encoder(%ctx: !cheddar.context) -> !cheddar.encoder {
  // expected-error @below {{lowering of 'cheddar.get_encoder' is not supported}}
  %e = cheddar.get_encoder %ctx : (!cheddar.context) -> !cheddar.encoder
  return %e : !cheddar.encoder
}

// -----

func.func @create_user_interface(%ctx: !cheddar.context)
    -> !cheddar.user_interface {
  // expected-error @below {{lowering of 'cheddar.create_user_interface' is not supported}}
  %ui = cheddar.create_user_interface %ctx
      : (!cheddar.context) -> !cheddar.user_interface
  return %ui : !cheddar.user_interface
}
