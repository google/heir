// RUN: heir-opt --cheddar-to-emitc --split-input-file --verify-diagnostics %s

// A dynamic-shape memref of a move-only cheddar type can't be represented as a
// fixed-size emitc.array, so the type converter refuses it and the conversion
// fails -- rather than falling through to a memref<?x!emitc.opaque> that stock
// MemRefToEmitC would lower with copies of move-only payloads.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @dynamic(%m: memref<?x!cheddar.ciphertext>, %i: index)
    -> !cheddar.ciphertext {
  %0 = memref.load %m[%i] : memref<?x!cheddar.ciphertext>
  return %0 : !cheddar.ciphertext
}

// -----

// A rank>1 memref of a move-only cheddar type would need a multi-dimensional
// emitc.array, which the destination-passing boundary lift (1-D std::array)
// can't represent, so the conversion fails rather than emit invalid C++.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @rank2(%m: memref<2x3x!cheddar.ciphertext>, %i: index, %j: index)
    -> !cheddar.ciphertext {
  %0 = memref.load %m[%i, %j] : memref<2x3x!cheddar.ciphertext>
  return %0 : !cheddar.ciphertext
}

// -----

// An scf.for that carries a move-only payload through an iter_arg can't be
// lowered: the loop-carried value would need a copy-initialised accumulator,
// which the move-only payload types forbid. Reject it up front (a real
// reduction writes into a buffer instead -- see @loop_store in compile/).
func.func @loop_carry(%ctx: !cheddar.context, %in: memref<8x!cheddar.ciphertext>,
                      %seed: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @below {{carrying a move-only Cheddar value through a loop-carried or result value is not supported}}
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> !cheddar.ciphertext {
    %e = memref.load %in[%i] : memref<8x!cheddar.ciphertext>
    %s = cheddar.add %ctx, %acc, %e
        : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
    scf.yield %s : !cheddar.ciphertext
  }
  return %r : !cheddar.ciphertext
}
