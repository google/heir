// RUN: heir-opt --cheddar-to-emitc --verify-diagnostics %s

// HRot/HRotAdd/HConj/HConjAdd discover the UserInterface from the enclosing
// function's argument list at lowering time. Functions that lack such an
// argument cannot be legalized.

func.func @hrot_without_ui(%ctx: !cheddar.context, %ct: !cheddar.ciphertext)
    -> !cheddar.ciphertext {
  // The conversion framework emits "failed to legalize"; whether the
  // pattern's own diagnostic ("enclosing function is missing UserInterface
  // arg") is also surfaced depends on MLIR's internal dispatch.
  // expected-error@+1 {{'cheddar.hrot'}}
  %r = cheddar.hrot %ctx, %ct {static_distance = 1 : i64}
      : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
