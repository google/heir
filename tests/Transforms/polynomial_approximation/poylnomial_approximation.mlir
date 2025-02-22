// RUN: heir-opt --polynomial-approximation %s | FileCheck %s

// CHECK-LABEL: @test_exp
func.func @test_exp(%x: f32) -> f32 {
  // CHECK: polynomial.eval
  // CHECK-SAME: 0.99458116404270657 + 0.99565537253615788x + 0.54297028147256321x**2 + 0.17954582110873779x**3
  %0 = math.exp %x : f32
  return %0 : f32
}
