// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s
// FIXME: with the addition of IndexType support, this test actually passes.
// However, the issue itself is actually about the unexpected segfault
// that occurred here when the typeconverter does NOT have a conversion for IndexType.
#alignment = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>
// CHECK: @foo
func.func @foo(%arg0: !secret.secret<i32> {tensor_ext.layout = #layout}) -> (!secret.secret<index> {tensor_ext.layout = #layout}) {
  %0 = secret.generic(%arg0: !secret.secret<i32> {tensor_ext.layout = #layout}) {
  ^body(%input0: i32):
    // CHECK: arith.index_cast
    %1 = arith.index_cast %input0 {tensor_ext.layout = #layout} : i32 to index
    secret.yield %1 : index
  } -> (!secret.secret<index> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<index>
}
