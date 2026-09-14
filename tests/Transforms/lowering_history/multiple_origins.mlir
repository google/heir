// RUN: not heir-opt %s --explain-lowering-history --polynomial-approximation 2>&1 | FileCheck %s

// CHECK: error: {{.*}}domain_lower must be strictly less than domain_upper
// CHECK-DAG: note: lowering history: 'torch.aten.exp' observed at stage 'left'
// CHECK-DAG: note: lowering history: 'torch.aten.add' observed at stage 'right'
#left = loc(fused<{heir.lowering_stage = "left"}>["torch.aten.exp"("left.py":1:1)])
#right = loc(fused<{heir.lowering_stage = "right"}>["torch.aten.add"("right.py":2:1)])
func.func @fused(%x: f32 {secret.secret}) -> f32 {
  %y = math.exp %x {domain_lower = 1.0 : f64, domain_upper = 0.0 : f64} : f32 loc(fused[#left, #right])
  return %y : f32
}
