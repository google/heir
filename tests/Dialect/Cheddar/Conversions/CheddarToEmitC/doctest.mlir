// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// A destination-passing `cheddar.add` lowers to `ctx->Add(out, a, b)`; the
// boundary pass re-types the payload args as `Ciphertext<word>&` (out-param)
// and `const Ciphertext<word>&` (inputs).

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context

// CHECK: func.func @add(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"Context<word>">>
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK: emitc.member_call_opaque %arg0 "Add"(%arg3, %arg1, %arg2)
func.func @add(%ctx: !context, %a: tensor<!ciphertext>,
               %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d = tensor.empty() : tensor<!ciphertext>
  %c = cheddar.add %ctx, %a, %b, %d
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %c : tensor<!ciphertext>
}
