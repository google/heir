// RUN: not heir-opt %s --validate-tensor-kernels --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: model.py:42:1: error: indexing an encrypted tensor with a runtime index is not supported
// CHECK: note: tensor.extract requires compile-time constant indices
// CHECK-NOT: failed to legalize
func.func @runtime_index(%x: tensor<4xf32> {secret.secret}, %i: index) -> f32 {
  %y = tensor.extract %x[%i] : tensor<4xf32> loc("embedding"("model.py":42:1))
  return %y : f32
}
