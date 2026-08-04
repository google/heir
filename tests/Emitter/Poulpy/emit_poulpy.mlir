// RUN: heir-translate %s --emit-poulpy | FileCheck %s


// CHECK: type BE = NTT4x30Ref;
// CHECK: type Ct = CKKSCiphertext<<BE as Backend>::OwnedBuf>;
!module = !poulpy.module<ntt4x30_ref>
!scratch   = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>
!ctu = memref<!poulpy.unnormalized_ciphertext>
!pt = memref<!poulpy.plaintext>
!sk = !poulpy.secret_key
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

// CHECK: pub fn add_unnormalized(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut CtUnnorm
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @add_unnormalized(%m: !module, %s: !scratch, %dst: !ctu, %a: !ct, %b: !ct) {
  // CHECK: [[m]].ckks_add_into_unnormalized(&mut *[[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add_unnormalized %m, %dst, %a, %b, %s : (!module, !ctu, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn add_unnormalized_alloc(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @add_unnormalized_alloc(%m: !module, %s: !scratch, %a: !ct, %b: !ct) {
  %dst = memref.alloc() : !ctu
  // CHECK: let mut [[dst:v[0-9]+]] = CtUnnorm::new([[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k()));
  // CHECK-NEXT: [[m]].ckks_add_into_unnormalized(&mut [[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add_unnormalized %m, %dst, %a, %b, %s : (!module, !ctu, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn sub_unnormalized(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut CtUnnorm
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @sub_unnormalized(%m: !module, %s: !scratch, %dst: !ctu, %a: !ct, %b: !ct) {
  // CHECK: [[m]].ckks_sub_into_unnormalized(&mut *[[dst]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.sub_unnormalized %m, %dst, %a, %b, %s : (!module, !ctu, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn normalize(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &CtUnnorm
// CHECK-NEXT: ) -> Result<Ct> {
func.func @normalize(%m: !module, %s: !scratch, %a: !ctu) -> !ct {
  %res = memref.alloc() : !ct
  // CHECK: let mut [[res:v[0-9]+]] = [[a]].clone().normalize(&*[[m]], &mut [[s]].borrow());
  poulpy.normalize %m, %res, %a, %s : (!module, !ct, !ctu, !scratch) -> ()
  // CHECK-NEXT: Ok([[res]])
  return %res : !ct
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

// CHECK: pub fn encode_check(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[re:v[0-9]+]]: &[f64]
// CHECK: [[im:v[0-9]+]]: &[f64]
// CHECK-NEXT: ) -> Result<()> {
func.func @encode_check(%m: !module, %re: memref<4xf64>, %im: memref<4xf64>) {
  %pt = memref.alloc() : !pt
  // CHECK: let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>([[m]].n() / 2)?;
  // CHECK-NEXT: let mut [[pt:v[0-9]+]] = [[m]].ckks_pt_vec_alloc(Base2K(52u32), TorusPrecision(65u32));
  // CHECK-NEXT: [[pt]].set_meta(CKKSMeta { log_delta: 45usize, log_sparsity: 0usize });
  // CHECK-NEXT: encoder.encode_reim(&mut [[pt]], &*[[re]], &*[[im]])?;
  poulpy.encode %m, %pt, %re, %im {logDelta = 45 : i64, logBudget = 20 : i64, base2k = 52 : i64} : (!module, !pt, memref<4xf64>, memref<4xf64>) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn decode_check(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt
// CHECK: [[re:v[0-9]+]]: &mut [f64]
// CHECK: [[im:v[0-9]+]]: &mut [f64]
// CHECK-NEXT: ) -> Result<()> {
func.func @decode_check(%m: !module, %pt: !pt, %re: memref<4xf64>, %im: memref<4xf64>) {
  // CHECK: let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>([[m]].n() / 2)?;
  // CHECK-NEXT: encoder.decode_reim(&*[[pt]], &mut *[[re]], &mut *[[im]])?;
  poulpy.decode %m, %re, %im, %pt : (!module, memref<4xf64>, memref<4xf64>, !pt) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn encrypt_check(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk
// CHECK: [[pt:v[0-9]+]]: &Pt
// CHECK-NEXT: ) -> Result<()> {
func.func @encrypt_check(%m: !module, %s: !scratch, %sk: !sk, %pt: !pt) {
  %ct = memref.alloc() : !ct
  // CHECK: let mut [[ct:v[0-9]+]] = [[m]].ckks_ciphertext_alloc(Base2K(52u32), TorusPrecision(300u32));
  // CHECK-NEXT: let enc_layout0 = EncryptionLayout::new_from_default_sigma(GLWELayout {
  // CHECK-NEXT: n: [[m]].ring_degree(), base2k: Base2K(52u32), k: TorusPrecision(300u32), rank: [[sk]].rank(),
  // CHECK-NEXT: })?;
  // CHECK-NEXT: let mut source0 = Source::new([0u8; 32]);
  // CHECK-NEXT: let mut source1 = Source::new([1u8; 32]);
  // CHECK-NEXT: [[m]].ckks_encrypt_sk(&mut [[ct]], &*[[pt]], &*[[sk]], &enc_layout0, &mut source0, &mut source1, &mut [[s]].borrow())?;
  poulpy.encrypt %m, %ct, %pt, %sk, %s {base2k = 52 : i64, ctk = 300 : i64} : (!module, !ct, !pt, !sk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn encrypt_twice_check(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk
// CHECK: [[pt:v[0-9]+]]: &Pt
// CHECK-NEXT: ) -> Result<()> {
func.func @encrypt_twice_check(%m: !module, %s: !scratch, %sk: !sk, %pt: !pt) {
  %ct0 = memref.alloc() : !ct
  // CHECK: let enc_layout0 = EncryptionLayout::new_from_default_sigma(GLWELayout {
  // CHECK-NEXT: n: [[m]].ring_degree(), base2k: Base2K(52u32), k: TorusPrecision(300u32), rank: [[sk]].rank(),
  // CHECK-NEXT: })?;
  // CHECK-NEXT: let mut source0 = Source::new([0u8; 32]);
  // CHECK-NEXT: let mut source1 = Source::new([1u8; 32]);
  poulpy.encrypt %m, %ct0, %pt, %sk, %s {base2k = 52 : i64, ctk = 300 : i64} : (!module, !ct, !pt, !sk, !scratch) -> ()
  %ct1 = memref.alloc() : !ct
  // CHECK: let enc_layout2 = EncryptionLayout::new_from_default_sigma(GLWELayout {
  // CHECK-NEXT: n: [[m]].ring_degree(), base2k: Base2K(52u32), k: TorusPrecision(300u32), rank: [[sk]].rank(),
  // CHECK-NEXT: })?;
  // CHECK-NEXT: let mut source2 = Source::new([2u8; 32]);
  // CHECK-NEXT: let mut source3 = Source::new([3u8; 32]);
  poulpy.encrypt %m, %ct1, %pt, %sk, %s {base2k = 52 : i64, ctk = 300 : i64} : (!module, !ct, !pt, !sk, !scratch) -> ()
  // CHECK: Ok(())
  return
}

// CHECK: pub fn decrypt_check(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk
// CHECK: [[ct:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<()> {
func.func @decrypt_check(%m: !module, %s: !scratch, %sk: !sk, %ct: !ct) {
  %pt = memref.alloc() : !pt
  // CHECK: let mut [[pt:v[0-9]+]] = [[m]].ckks_pt_vec_alloc_from_infos(&*[[ct]]);
  // CHECK-NEXT: [[m]].ckks_decrypt(&mut [[pt]], &*[[ct]], &*[[sk]], &mut [[s]].borrow())?;
  poulpy.decrypt %m, %pt, %ct, %sk, %s : (!module, !pt, !ct, !sk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn two_results(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<(Ct, Ct)> {
func.func @two_results(%mod: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  %sum = memref.alloc() : !ct
  // CHECK: let mut [[sum:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
  // CHECK-NEXT: [[m]].ckks_add_into(&mut [[sum]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add %mod, %sum, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: Ok(([[sum]], [[a]].clone()))
  return %sum, %a : !ct, !ct
}

// CHECK: pub fn square(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[dst:v[0-9]+]]: &mut Ct
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[tsk:v[0-9]+]]: &Tsk
// CHECK-NEXT: ) -> Result<()> {
func.func @square(%m: !module, %s: !scratch, %dst: !ct, %a: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_mul_into(&mut *[[dst]], &*[[a]], &*[[a]], &*[[tsk]], &mut [[s]].borrow())?;
  poulpy.mul %m, %dst, %a, %a, %tsk, %s : (!module, !ct, !ct, !ct, !tsk, !scratch) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn call_check(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &mut Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK: [[tsk:v[0-9]+]]: &Tsk
// CHECK-NEXT: ) -> Result<()> {
func.func @call_check(%mod: !module, %s: !scratch, %a: !ct, %b: !ct, %tsk: !tsk) {
  // CHECK: [[m]].ckks_add_assign(&mut *[[a]], &*[[b]], &mut [[s]].borrow())?;
  poulpy.add_assign %mod, %a, %b, %s : (!module, !ct, !ct, !scratch) -> ()
  // CHECK-NEXT: square(&*[[m]], &mut *[[s]], &mut *[[a]], &*[[b]], &*[[tsk]])?;
  func.call @square(%mod, %s, %a, %b, %tsk) : (!module, !scratch, !ct, !ct, !tsk) -> ()
  // CHECK-NEXT: Ok(())
  return
}

// CHECK: pub fn get_one(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<Ct> {
func.func @get_one(%m: !module, %s: !scratch, %a: !ct) -> !ct {
  // CHECK: Ok([[a]].clone())
  return %a : !ct
}

// CHECK: pub fn call_one_result(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<Ct> {
func.func @call_one_result(%mod: !module, %s: !scratch, %a: !ct) -> !ct {
  // CHECK: let [[r:v[0-9]+]] = get_one(&*[[m]], &mut *[[s]], &*[[a]])?;
  %r = func.call @get_one(%mod, %s, %a) : (!module, !scratch, !ct) -> !ct
  // CHECK-NEXT: Ok([[r]])
  return %r : !ct
}

// CHECK: pub fn get_two(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<(Ct, Ct)> {
func.func @get_two(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  // CHECK: Ok(([[a]].clone(), [[b]].clone()))
  return %a, %b : !ct, !ct
}

// CHECK: pub fn call_two_results(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct
// CHECK: [[b:v[0-9]+]]: &Ct
// CHECK-NEXT: ) -> Result<(Ct, Ct)> {
func.func @call_two_results(%mod: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  // CHECK: let ([[r0:v[0-9]+]], [[r1:v[0-9]+]]) = get_two(&*[[m]], &mut *[[s]], &*[[a]], &*[[b]])?;
  %r0, %r1 = func.call @get_two(%mod, %s, %a, %b) : (!module, !scratch, !ct, !ct) -> (!ct, !ct)
  // CHECK-NEXT: Ok(([[r0]], [[r1]]))
  return %r0, %r1 : !ct, !ct
}
