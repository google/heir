// RUN: heir-translate %s --emit-poulpy | FileCheck %s


// CHECK: type BE = NTT4x30Ref;
// CHECK: type Ct = CKKSCiphertext<<BE as Backend>::OwnedBuf>;
!module = !poulpy.module<ntt4x30_ref>
!scratch   = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>
!tsk = !poulpy.tensor_key
!akm = !poulpy.automorphism_key_map

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
// CHECK: pub fn mul(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK: [[tsk:v[0-9]+]]: &Tsk
// CHECK-NEXT: ) -> Result<()> {
func.func @mul(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %b: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_mul_into(&mut *[[dst]], &*[[a]], &*[[b]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %m, %dst, %a, %b, %tsk, %s : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_assign(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[tsk:v[0-9]+]]: &Tsk
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_assign(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_mul_assign(&mut *[[dst]], &*[[a]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul_assign %m, %dst, %a, %tsk, %s : (!module, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_add(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_add(%mod: !module, %scratch: !scratch, %tsk: !tsk, %a: !ct, %b: !ct) {
  %sum = memref.alloc() : !ct
  // CHECK: let mut [[sum:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
  // CHECK-NEXT: [[m]].ckks_add_into(&mut [[sum]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add %mod, %sum, %a, %b, %scratch : (!module, !ct, !ct, !ct, !scratch) -> ()
  %prod = memref.alloc() : !ct
  // CHECK-NEXT: let mut [[prod:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[sum]].base2k(), [[sum]].max_k());
  // CHECK-NEXT: [[m]].ckks_mul_into(&mut [[prod]], &[[sum]], &*[[b]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %mod, %prod, %sum, %b, %tsk, %scratch : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: [[m]].ckks_mul_into(&mut [[prod]], &[[sum]], &*[[b]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %mod, %prod, %sum, %b, %tsk, %scratch : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rotate(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[src:v[0-9]+]]: &Ct
// CHECK: [[akm:v[0-9]+]]: &Akm
// CHECK-NEXT: ) -> Result<()> {
func.func @rotate(%m: !module, %s: !scratch, %dst: !ct, %src: !ct, %akm: !akm) {
  // CHECK: [[m]].ckks_rotate_into(&mut *[[dst]], &*[[src]], 1i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate %m, %dst, %src, %akm, %s {k = 1 : i64} : (!module, !ct, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rotate_assign(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[akm:v[0-9]+]]: &Akm
// CHECK-NEXT: ) -> Result<()> {
func.func @rotate_assign(%m: !module, %s: !scratch, %dst: !ct, %akm: !akm) {
  // CHECK: [[m]].ckks_rotate_assign(&mut *[[dst]], 1i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate_assign %m, %dst, %akm, %s {k = 1 : i64} : (!module, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rotate_alloc(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct
// CHECK: [[akm:v[0-9]+]]: &Akm
// CHECK-NEXT: ) -> Result<()> {
func.func @rotate_alloc(%m: !module, %s: !scratch, %src: !ct, %akm: !akm) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].max_k());
  // CHECK-NEXT: [[m]].ckks_rotate_into(&mut [[dst]], &*[[src]], 2i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate %m, %dst, %src, %akm, %s {k = 2 : i64} : (!module, !ct, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rot_mul_add(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk
// CHECK: [[akm:v[0-9]+]]: &Akm
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<Ct> {
func.func @rot_mul_add(%mod: !module, %scratch: !scratch, %tsk: !tsk, %akm: !akm,
                       %a: !ct, %b: !ct) -> !ct {
  %sum = memref.alloc() : !ct
  // CHECK: let mut [[sum:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
  // CHECK-NEXT: [[m]].ckks_add_into(&mut [[sum]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add %mod, %sum, %a, %b, %scratch
      : (!module, !ct, !ct, !ct, !scratch) -> ()
  %prod = memref.alloc() : !ct
  // CHECK-NEXT: let mut [[prod:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[sum]].base2k(), [[sum]].max_k());
  // CHECK-NEXT: [[m]].ckks_mul_into(&mut [[prod]], &[[sum]], &*[[b]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %mod, %prod, %sum, %b, %tsk, %scratch
      : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: [[m]].ckks_rotate_assign(&mut [[prod]], 1i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate_assign %mod, %prod, %akm, %scratch {k = 1 : i64}
      : (!module, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok([[prod]])
  // CHECK-NEXT: }
  return %prod : !ct
}

// CHECK: pub fn rescale(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[src:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @rescale(%m: !module, %s: !scratch, %dst: !ct, %src: !ct) {
  // CHECK: [[m]].ckks_div_pow2_into(&mut *[[dst]], &*[[src]], 3usize, &mut [[s]].borrow())?;
  poulpy.rescale %m, %dst, %src, %s {bits = 3 : i64} : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rescale_assign(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @rescale_assign(%m: !module, %s: !scratch, %dst: !ct) {
  // CHECK: [[m]].ckks_div_pow2_assign(&mut *[[dst]], 3usize)?;
  poulpy.rescale_assign %m, %dst, %s {bits = 3 : i64} : (!module, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rescale_alloc(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @rescale_alloc(%m: !module, %s: !scratch, %src: !ct) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].max_k());
  // CHECK-NEXT: [[m]].ckks_div_pow2_into(&mut [[dst]], &*[[src]], 4usize, &mut [[s]].borrow())?;
  poulpy.rescale %m, %dst, %src, %s {bits = 4 : i64} : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn compact_limbs(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[src:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @compact_limbs(%m: !module, %s: !scratch, %dst: !ct, %src: !ct) {
  // CHECK: [[m]].ckks_copy(&mut *[[dst]], &*[[src]], &mut [[s]].borrow())?;
  poulpy.compact_limbs %m, %dst, %src, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn compact_limbs_alloc(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @compact_limbs_alloc(%m: !module, %s: !scratch, %src: !ct) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].k());
  // CHECK-NEXT: [[m]].ckks_copy(&mut [[dst]], &*[[src]], &mut [[s]].borrow())?;
  poulpy.compact_limbs %m, %dst, %src, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}
