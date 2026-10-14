// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --cheddar-emitc-boundary --verify-diagnostics %s

// A payload loaded from a buffer is an lvalue behind a materialization cast.
// Uses the lowering does not pass the lvalue to directly would keep that cast,
// which --reconcile-unrealized-casts cannot remove and the C++ emitter cannot
// print, so the boundary pass rejects them.
func.func @return_eval_key(%keys: memref<2x!cheddar.eval_key>, %i: index) -> !cheddar.eval_key {
  %k = memref.load %keys[%i] : memref<2x!cheddar.eval_key>
  // expected-error @below {{converts an emitc.lvalue for a use the Cheddar EmitC lowering does not support}}
  return %k : !cheddar.eval_key
}
