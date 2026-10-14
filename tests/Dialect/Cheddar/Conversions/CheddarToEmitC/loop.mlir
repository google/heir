// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// A loop kernel: the payload producer in the scf.for body writes `out[i]`
// directly; no payload copy survives bufferization.

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context

// The ops inside the `emitc.for` body print without the `emitc.` prefix (emitc
// is the body region's default dialect), so match the bare op names there.
// CHECK: func.func @loop_store
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: !emitc.array<8x!emitc.opaque<"Ciphertext<word>">> {bufferize.result}
// CHECK-NOT: emitc.variable
// CHECK: emitc.for
// CHECK: %[[ELEMENT:.*]] = subscript %[[OUT]][%{{.*}}]
// CHECK: member_call_opaque %{{.*}} "Add"(%[[ELEMENT]],
func.func @loop_store(%ctx: !context, %in: tensor<!ciphertext>)
    -> tensor<8x!ciphertext> {
  %out = tensor.empty() : tensor<8x!ciphertext>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %out)
      -> (tensor<8x!ciphertext>) {
    %d = tensor.empty() : tensor<!ciphertext>
    %v = cheddar.add %ctx, %in, %in, %d
        : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>)
        -> tensor<!ciphertext>
    %ins = tensor.insert_slice %v into %acc[%i] [1] [1]
        : tensor<!ciphertext> into tensor<8x!ciphertext>
    scf.yield %ins : tensor<8x!ciphertext>
  }
  return %r : tensor<8x!ciphertext>
}
