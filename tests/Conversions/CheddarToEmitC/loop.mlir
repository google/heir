// RUN: heir-opt --cheddar-to-emitc --reconcile-unrealized-casts %s | FileCheck %s

// A bufferized loop kernel that writes ciphertexts into an output buffer.
// SCF and Arith are lowered to EmitC in the same conversion as the cheddar
// ops (no separate per-dialect EmitC passes), so the whole loop lowers under a
// single --cheddar-to-emitc invocation: the index induction variable becomes
// emitc.size_t and the move-only memref.store becomes a subscript + std::move.
//
// The output memref is written, so its boundary type is a *mutable*
// std::array reference (a const one would reject the std::move store).

// The emitc ops inside the emitc.for body print without the `emitc.` prefix
// (emitc is the body region's default dialect), so match the bare op names.
// CHECK: func.func @loop_store
// CHECK-SAME: !emitc.opaque<"std::array<Ciphertext<word>, 8>&">
// CHECK-NOT: const std::array
// CHECK: emitc.for
// CHECK: subscript %arg2
// CHECK: verbatim "{} = std::move({});"
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
