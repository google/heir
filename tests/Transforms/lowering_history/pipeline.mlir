// RUN: not heir-opt %s --torch-linalg-to-ckks=lowering-history=true --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: error: 'math.tanh' op domain_lower must be strictly less than domain_upper
// CHECK: note: lowering history: 'math.tanh' observed at stage 'after-linalg-preprocessing'
// CHECK: note: lowering history: 'math.tanh' observed at stage 'torch-linalg-input'
func.func @bad_domain(%x: f32 {secret.secret}) -> f32 {
  %y = math.tanh %x {domain_lower = 1.0 : f64, domain_upper = 0.0 : f64} : f32 loc("model.py":42:1)
  return %y : f32
}
