// Each pass of the lowering runs on its own. --convert-to-emitc only needs the
// dialects in filter-dialects to be loaded (here by the input); the boundary
// pass only creates EmitC ops.
// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar %s | FileCheck %s --check-prefix=CONVERT
// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar %s | heir-opt --cheddar-emitc-boundary | FileCheck %s --check-prefix=BOUNDARY

// CONVERT-NOT: emitc.include
// CONVERT: func.func @neg(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>, %[[IN:.*]]: !emitc.lvalue<!emitc.opaque<"Ciphertext<word>">>, %[[OUT:.*]]: !emitc.lvalue<!emitc.opaque<"Ciphertext<word>">>)
// CONVERT: emitc.member_call_opaque %[[CTX]] "Neg"(%[[OUT]], %[[IN]]) {cheddar.destination_operand = 1 : i64}

// BOUNDARY: emitc.include "core/Context.h"
// BOUNDARY: func.func @neg(%{{.*}}: !emitc.ptr<!emitc.opaque<"Context<word>">>, %{{.*}}: !emitc.opaque<"const Ciphertext<word>&">, %{{.*}}: !emitc.opaque<"Ciphertext<word>&">)
// BOUNDARY-NOT: cheddar.destination_operand
func.func @neg(%ctx: !cheddar.context, %in: memref<!cheddar.ciphertext>,
               %out: memref<!cheddar.ciphertext>) {
  cheddar.neg %ctx, %in, %out
      : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  return
}
