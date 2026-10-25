// RUN: heir-translate %s --emit-poulpy | FileCheck %s


// CHECK: pub trait PoulpyBackend:
// CHECK: type Ct<BE> = CKKSCiphertextOwned<BE>;
!module = !poulpy.module
!scratch   = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>
!pt = memref<!poulpy.plaintext>
!sk = !poulpy.secret_key
!tsk = !poulpy.tensor_key
!akm = !poulpy.automorphism_key_map

// CHECK: pub fn f<BE: PoulpyBackend>(
// CHECK: [[v:v[0-9]+]]: &Module<BE>
// CHECK: [[v:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK-NEXT: ) -> Result<()> {
// CHECK: Ok(())
// CHECK-NEXT: }
func.func @f(%m: !module, %s: !scratch) {
  return
}

// CHECK: [[v:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>>
func.func @passthrough(%a: !ct) -> !ct {
  // CHECK: Ok([[v]].clone())
  return %a : !ct
}

// CHECK: pub fn add<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @add(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %b: !ct) {
  // CHECK: [[m]].ckks_add_into(&mut *[[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add %m, %dst, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @sub(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %b: !ct) {
  // CHECK: [[m]].ckks_sub_into(&mut *[[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.sub %m, %dst, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @sub_assign(%m: !module, %s: !scratch, %dst: !ct, %a: !ct) {
  // CHECK: [[m]].ckks_sub_assign(&mut *[[dst]], &*[[a]], &mut [[s]].borrow())?;
  poulpy.sub_assign %m, %dst, %a, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}
// CHECK: pub fn mul<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %b: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_mul_into(&mut *[[dst]], &*[[a]], &*[[b]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %m, %dst, %a, %b, %tsk, %s : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_assign(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_mul_assign(&mut *[[dst]], &*[[a]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul_assign %m, %dst, %a, %tsk, %s : (!module, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn add_pt_vec<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @add_pt_vec(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %pt: !pt) {
  // CHECK: [[m]].ckks_add_pt_vec_into(&mut *[[dst]], &*[[a]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.add_pt_vec %m, %dst, %a, %pt, %s : (!module, !ct, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn add_pt_vec_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @add_pt_vec_assign(%m: !module, %s: !scratch, %dst: !ct, %pt: !pt) {
  // CHECK: [[m]].ckks_add_pt_vec_assign(&mut *[[dst]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.add_pt_vec_assign %m, %dst, %pt, %s : (!module, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub_pt_vec<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @sub_pt_vec(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %pt: !pt) {
  // CHECK: [[m]].ckks_sub_pt_vec_into(&mut *[[dst]], &*[[a]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.sub_pt_vec %m, %dst, %a, %pt, %s : (!module, !ct, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub_pt_vec_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @sub_pt_vec_assign(%m: !module, %s: !scratch, %dst: !ct, %pt: !pt) {
  // CHECK: [[m]].ckks_sub_pt_vec_assign(&mut *[[dst]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.sub_pt_vec_assign %m, %dst, %pt, %s : (!module, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_pt_vec<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_pt_vec(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %pt: !pt) {
  // CHECK: [[m]].ckks_mul_pt_vec_into(&mut *[[dst]], &*[[a]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.mul_pt_vec %m, %dst, %a, %pt, %s : (!module, !ct, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_pt_vec_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_pt_vec_assign(%m: !module, %s: !scratch, %dst: !ct, %pt: !pt) {
  // CHECK: [[m]].ckks_mul_pt_vec_assign(&mut *[[dst]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.mul_pt_vec_assign %m, %dst, %pt, %s : (!module, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// The destination is sized from the ciphertext operand, not the plaintext.
// CHECK: pub fn mul_pt_vec_alloc<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
func.func @mul_pt_vec_alloc(%m: !module, %s: !scratch, %a: !ct, %pt: !pt) -> !ct {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
  // CHECK-NEXT: [[m]].ckks_mul_pt_vec_into(&mut [[dst]], &*[[a]], &*[[pt]], &mut [[s]].borrow())?;
  poulpy.mul_pt_vec %m, %dst, %a, %pt, %s : (!module, !ct, !ct, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok([[dst]])
  return %dst : !ct
}

// CHECK: pub fn neg<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @neg(%m: !module, %s: !scratch, %dst: !ct, %src: !ct) {
  // CHECK: [[m]].ckks_neg_into(&mut *[[dst]], &*[[src]], &mut [[s]].borrow())?;
  poulpy.neg %m, %dst, %src, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn neg_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @neg_assign(%m: !module, %s: !scratch, %dst: !ct) {
  // CHECK: [[m]].ckks_neg_assign(&mut *[[dst]])?;
  poulpy.neg_assign %m, %dst, %s : (!module, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn neg_alloc<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
func.func @neg_alloc(%m: !module, %s: !scratch, %src: !ct) -> !ct {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].max_k());
  // CHECK-NEXT: [[m]].ckks_neg_into(&mut [[dst]], &*[[src]], &mut [[s]].borrow())?;
  poulpy.neg %m, %dst, %src, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok([[dst]])
  return %dst : !ct
}

// CHECK: pub fn mul_add<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
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

// CHECK: pub fn rotate<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK: [[akm:v[0-9]+]]: &Akm<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @rotate(%m: !module, %s: !scratch, %dst: !ct, %src: !ct, %akm: !akm) {
  // CHECK: [[m]].ckks_rotate_into(&mut *[[dst]], &*[[src]], 1i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate %m, %dst, %src, %akm, %s <k = 1> : (!module, !ct, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rotate_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[akm:v[0-9]+]]: &Akm<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @rotate_assign(%m: !module, %s: !scratch, %dst: !ct, %akm: !akm) {
  // CHECK: [[m]].ckks_rotate_assign(&mut *[[dst]], 1i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate_assign %m, %dst, %akm, %s <k = 1> : (!module, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rotate_alloc<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK: [[akm:v[0-9]+]]: &Akm<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @rotate_alloc(%m: !module, %s: !scratch, %src: !ct, %akm: !akm) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].max_k());
  // CHECK-NEXT: [[m]].ckks_rotate_into(&mut [[dst]], &*[[src]], 2i64, &*[[akm]], &mut [[s]].borrow())?;
  poulpy.rotate %m, %dst, %src, %akm, %s <k = 2> : (!module, !ct, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn rot_mul_add<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK: [[akm:v[0-9]+]]: &Akm<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
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
  poulpy.rotate_assign %mod, %prod, %akm, %scratch <k = 1>
      : (!module, !ct, !akm, !scratch) -> ()
  // CHECK-NEXT: Ok([[prod]])
  // CHECK-NEXT: }
  return %prod : !ct
}

// CHECK: pub fn mul_pow2<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_pow2(%m: !module, %s: !scratch, %dst: !ct, %src: !ct) {
  // CHECK: [[m]].ckks_mul_pow2_into(&mut *[[dst]], &*[[src]], 3usize, &mut [[s]].borrow())?;
  poulpy.mul_pow2 %m, %dst, %src, %s <bits = 3> : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_pow2_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_pow2_assign(%m: !module, %s: !scratch, %dst: !ct) {
  // CHECK: [[m]].ckks_mul_pow2_assign(&mut *[[dst]], 3usize, &mut [[s]].borrow())?;
  poulpy.mul_pow2_assign %m, %dst, %s <bits = 3> : (!module, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn mul_pow2_alloc<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @mul_pow2_alloc(%m: !module, %s: !scratch, %src: !ct) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].max_k());
  // CHECK-NEXT: [[m]].ckks_mul_pow2_into(&mut [[dst]], &*[[src]], 4usize, &mut [[s]].borrow())?;
  poulpy.mul_pow2 %m, %dst, %src, %s <bits = 4> : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn div_pow2<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @div_pow2(%m: !module, %s: !scratch, %dst: !ct, %src: !ct) {
  // CHECK: [[m]].ckks_div_pow2_into(&mut *[[dst]], &*[[src]], 3usize, &mut [[s]].borrow())?;
  poulpy.div_pow2 %m, %dst, %src, %s <bits = 3> : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn div_pow2_assign<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @div_pow2_assign(%m: !module, %s: !scratch, %dst: !ct) {
  // CHECK: [[m]].ckks_div_pow2_assign(&mut *[[dst]], 3usize)?;
  poulpy.div_pow2_assign %m, %dst, %s <bits = 3> : (!module, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn div_pow2_alloc<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @div_pow2_alloc(%m: !module, %s: !scratch, %src: !ct) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].max_k());
  // CHECK-NEXT: [[m]].ckks_div_pow2_into(&mut [[dst]], &*[[src]], 4usize, &mut [[s]].borrow())?;
  poulpy.div_pow2 %m, %dst, %src, %s <bits = 4> : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn compact_limbs<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @compact_limbs(%m: !module, %s: !scratch, %dst: !ct, %src: !ct) {
  // CHECK: [[m]].ckks_copy(&mut *[[dst]], &*[[src]], &mut [[s]].borrow())?;
  poulpy.compact_limbs %m, %dst, %src, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn compact_limbs_alloc<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[src:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @compact_limbs_alloc(%m: !module, %s: !scratch, %src: !ct) {
  %dst = memref.alloc() : !ct
  // CHECK: let mut [[dst:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[src]].base2k(), [[src]].k());
  // CHECK-NEXT: [[m]].ckks_copy(&mut [[dst]], &*[[src]], &mut [[s]].borrow())?;
  poulpy.compact_limbs %m, %dst, %src, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn f64_check(
// CHECK: [[r:v[0-9]+]]: &[f64]
// CHECK-NEXT: ) -> Result<()> {
func.func @f64_check(%r: memref<8xf64>) {
  %tmp = memref.alloc() : memref<4xf64>
  // CHECK: let mut [[tmp:v[0-9]+]] = vec![0f64; 4];
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn encode_check<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[re:v[0-9]+]]: &[f64]
// CHECK: [[im:v[0-9]+]]: &[f64]
// CHECK-NEXT: ) -> Result<()> {
func.func @encode_check(%m: !module, %s: !scratch, %re: memref<4xf64>, %im: memref<4xf64>) {
  %pt = memref.alloc() : !pt
  // CHECK: let mut [[pt:v[0-9]+]] = [[m]].ckks_pt_vec_alloc(Base2K(52u32), TorusPrecision(65u32));
  // CHECK-NEXT: [[pt]].set_meta(CKKSMeta { log_delta: 45usize, log_sparsity: 0usize, slots: SlotsKind::Complex });
  // CHECK-NEXT: [[m]].ckks_encode_reim_into(&mut [[pt]], &*[[re]], &*[[im]], &mut [[s]].borrow())?;
  poulpy.encode %m, %pt, %re, %im, %s <logDelta = 45, logBudget = 20, base2k = 52> : (!module, !pt, memref<4xf64>, memref<4xf64>, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn decode_check<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK: [[re:v[0-9]+]]: &mut [f64]
// CHECK: [[im:v[0-9]+]]: &mut [f64]
// CHECK-NEXT: ) -> Result<()> {
func.func @decode_check(%m: !module, %s: !scratch, %pt: !pt, %re: memref<4xf64>, %im: memref<4xf64>) {
  // CHECK: [[m]].ckks_decode_reim_into(&*[[pt]], &mut *[[re]], &mut *[[im]], &mut [[s]].borrow())?;
  poulpy.decode %m, %re, %im, %pt, %s : (!module, memref<4xf64>, memref<4xf64>, !pt, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn encrypt_check<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @encrypt_check(%m: !module, %s: !scratch, %sk: !sk, %pt: !pt) {
  %ct = memref.alloc() : !ct
  // CHECK: let mut [[ct:v[0-9]+]] = [[m]].ckks_ciphertext_alloc(Base2K(52u32), TorusPrecision(300u32));
  // CHECK-NEXT: let mut source0 = Source::new([0u8; 32]);
  // CHECK-NEXT: let mut source1 = Source::new([1u8; 32]);
  // CHECK-NEXT: [[m]].ckks_encrypt_sk(&mut [[ct]], &*[[pt]], &*[[sk]], &mut source0, &mut source1, &mut [[s]].borrow())?;
  poulpy.encrypt %m, %ct, %pt, %sk, %s <base2k = 52, ctk = 300> : (!module, !ct, !pt, !sk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn encrypt_twice_check<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @encrypt_twice_check(%m: !module, %s: !scratch, %sk: !sk, %pt: !pt) {
  %ct0 = memref.alloc() : !ct
  // CHECK: let mut source0 = Source::new([0u8; 32]);
  // CHECK-NEXT: let mut source1 = Source::new([1u8; 32]);
  poulpy.encrypt %m, %ct0, %pt, %sk, %s <base2k = 52, ctk = 300> : (!module, !ct, !pt, !sk, !scratch) -> ()
  %ct1 = memref.alloc() : !ct
  // CHECK: let mut source2 = Source::new([2u8; 32]);
  // CHECK-NEXT: let mut source3 = Source::new([3u8; 32]);
  poulpy.encrypt %m, %ct1, %pt, %sk, %s <base2k = 52, ctk = 300> : (!module, !ct, !pt, !sk, !scratch) -> ()
  // CHECK: Ok(())
  return
}

// CHECK: pub fn decrypt_check<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk<BE>
// CHECK: [[ct:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @decrypt_check(%m: !module, %s: !scratch, %sk: !sk, %ct: !ct) {
  %pt = memref.alloc() : !pt
  // CHECK: let mut [[pt:v[0-9]+]] = [[m]].ckks_plaintext_alloc_from_infos(&*[[ct]]);
  // CHECK-NEXT: [[m]].ckks_decrypt(&mut [[pt]], &*[[ct]], &*[[sk]], &mut [[s]].borrow())?;
  poulpy.decrypt %m, %pt, %ct, %sk, %s : (!module, !pt, !ct, !sk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn two_results<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<(Ct<BE>, Ct<BE>)> {
func.func @two_results(%mod: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  %sum = memref.alloc() : !ct
  // CHECK: let mut [[sum:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
  // CHECK-NEXT: [[m]].ckks_add_into(&mut [[sum]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add %mod, %sum, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(([[sum]], [[a]].clone()))
  return %sum, %a : !ct, !ct
}

// CHECK: pub fn square<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @square(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_mul_into(&mut *[[dst]], &*[[a]], &*[[a]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %m, %dst, %a, %a, %tsk, %s : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn call_check<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK-NEXT: ) -> Result<()> {
func.func @call_check(%mod: !module, %s: !scratch, %a: !ct, %b: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_add_assign(&mut *[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add_assign %mod, %a, %b, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: square(&*[[m]], &mut *[[s]], &mut *[[a]], &*[[b]], &*[[tsk]])?;
  func.call @square(%mod, %s, %a, %b, %tsk) : (!module, !scratch, !ct, !ct, !tsk) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn get_one<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
func.func @get_one(%m: !module, %s: !scratch, %a: !ct) -> !ct {
  // CHECK: Ok([[a]].clone())
  return %a : !ct
}

// CHECK: pub fn call_one_result<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
func.func @call_one_result(%mod: !module, %s: !scratch, %a: !ct) -> !ct {
  // CHECK: let [[r:v[0-9]+]] = get_one(&*[[m]], &mut *[[s]], &*[[a]])?;
  %r = func.call @get_one(%mod, %s, %a) : (!module, !scratch, !ct) -> !ct
  // CHECK-NEXT: Ok([[r]])
  return %r : !ct
}

// CHECK: pub fn get_two<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<(Ct<BE>, Ct<BE>)> {
func.func @get_two(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  // CHECK: Ok(([[a]].clone(), [[b]].clone()))
  return %a, %b : !ct, !ct
}

// CHECK: pub fn call_two_results<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<(Ct<BE>, Ct<BE>)> {
func.func @call_two_results(%mod: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  // CHECK: let ([[r0:v[0-9]+]], [[r1:v[0-9]+]]) = get_two(&*[[m]], &mut *[[s]], &*[[a]], &*[[b]])?;
  %r0, %r1 = func.call @get_two(%mod, %s, %a, %b) : (!module, !scratch, !ct, !ct) -> (!ct, !ct)
  // CHECK-NEXT: Ok(([[r0]], [[r1]]))
  return %r0, %r1 : !ct, !ct
}

// CHECK: pub fn setup<BE: PoulpyBackend>(
// CHECK-NEXT: ) -> Result<(Module<BE>, ScratchOwned<BE>)> {
func.func @setup() -> (!module, !scratch) {
  // CHECK: let [[m:v[0-9]+]] = Module::<BE>::new(64u64);
  %mod = poulpy.module_create <N = 64> : () -> !module
  // CHECK-NEXT: let mut [[s:v[0-9]+]] = ScratchOwned::<BE>::alloc(1024usize);
  %scratch = poulpy.scratch_alloc <size = 1024> : () -> !scratch
  // CHECK-NEXT: Ok(([[m]], [[s]]))
  return %mod, %scratch : !module, !scratch
}
