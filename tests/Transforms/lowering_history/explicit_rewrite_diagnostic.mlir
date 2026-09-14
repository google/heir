// RUN: not heir-opt %s --explain-lowering-history --polynomial-approximation 2>&1 | FileCheck %s
// CHECK: error: 'math.exp' op domain_lower must be strictly less than domain_upper
// CHECK: note: lowering history: 'torch.aten.exp' lowered to 'math.exp' by --torch-to-linalg
#origin = loc(fused<{heir.lowering_pass = "torch-to-linalg", heir.lowering_result = "math.exp"}>["torch.aten.exp"("model.py":42:1)])
func.func @recorded_rewrite(%x: f32 {secret.secret}) -> f32 {
  %y = math.exp %x {domain_lower = 1.0 : f64, domain_upper = 0.0 : f64} : f32 loc(#origin)
  return %y : f32
}
