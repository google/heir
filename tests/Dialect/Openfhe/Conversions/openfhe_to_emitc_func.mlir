// RUN: heir-opt --convert-to-emitc %s | FileCheck %s

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext

module {
  // CHECK: emitc.func @dot_product__preprocessing(%[[ARG_CC:.*]]: !emitc.opaque<"CryptoContextT">) -> !emitc.ptr<!emitc.opaque<"Plaintext">>
  func.func @dot_product__preprocessing(%cc: !cc) -> memref<1x!pt> {
    %c0 = arith.constant 0 : index
    %alloc_0 = memref.alloc() : memref<8192xi64>
    // CHECK: %[[VEC:.*]] = call_opaque "std::vector<int64_t>"
    // CHECK: member_call_opaque %[[ARG_CC]] "MakePackedPlaintext"(%[[VEC]]) : !emitc.opaque<"CryptoContextT">, (!emitc.opaque<"std::vector<int64_t>">) -> !emitc.opaque<"Plaintext">
    %pt = openfhe.make_packed_plaintext %cc, %alloc_0 : (!cc, memref<8192xi64>) -> !pt
    %alloc = memref.alloc() : memref<1x!pt>
    memref.store %pt, %alloc[%c0] : memref<1x!pt>
    return %alloc : memref<1x!pt>
  }

  // CHECK: emitc.func @dot_product(%[[ARG_CC:.*]]: !emitc.opaque<"CryptoContextT">) {
  // CHECK: %[[RES:.*]] = call @dot_product__preprocessing(%[[ARG_CC]]) : (!emitc.opaque<"CryptoContextT">) -> !emitc.ptr<!emitc.opaque<"Plaintext">>
  func.func @dot_product(%cc: !cc) {
    %0 = func.call @dot_product__preprocessing(%cc) : (!cc) -> memref<1x!pt>
    %cast = memref.cast %0 : memref<1x!pt> to memref<1x!pt, strided<[?], offset: ?>>
    return
  }
}
