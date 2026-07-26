// RUN: heir-translate %s --emit-poulpy | FileCheck %s


// CHECK: type BE = NTT4x30Ref
!module = !poulpy.module<ntt4x30_ref>
// CHECK: pub fn f(
// CHECK: [[v:v[0-9]+]]: &Module<BE>,
// CHECK-NEXT: ) -> Result<()> {
// CHECK: Ok(())
// CHECK-NEXT: }
func.func @f(%m: !module) {
  return
}