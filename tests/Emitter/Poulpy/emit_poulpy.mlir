// RUN: heir-translate %s --emit-poulpy | FileCheck %s


// CHECK: type BE = NTT4x30Ref;
// CHECK: type Ct = CKKSCiphertext<<BE as Backend>::OwnedBuf>;
!module = !poulpy.module<ntt4x30_ref>
!scratch   = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>
// CHECK: pub fn f(
// CHECK: [[v:v[0-9]+]]: &Module<BE>
// CHECK: [[v:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK-NEXT: ) -> Result<()> {
// CHECK: Ok(())
// CHECK-NEXT: }
func.func @f(%m: !module, %s: !scratch) {
  return
}

// CHECK: [[v:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<Ct>
func.func @passthrough(%a: !ct) -> !ct {
  // CHECK: Ok([[v]].clone())
  return %a : !ct
}

// CHECK: pub fn add(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @add(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %b: !ct) {
  // CHECK: [[m]].ckks_add_into(&mut *[[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add %m, %dst, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @sub(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %b: !ct) {
  // CHECK: [[m]].ckks_sub_into(&mut *[[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.sub %m, %dst, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub_assign(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @sub_assign(%m: !module, %s: !scratch, %dst: !ct, %a: !ct) {
  // CHECK: [[m]].ckks_sub_assign(&mut *[[dst]], &*[[a]], &mut [[s]].borrow())?;
  poulpy.sub_assign %m, %dst, %a, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}
