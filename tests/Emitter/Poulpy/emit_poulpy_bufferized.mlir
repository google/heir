// RUN: heir-opt --poulpy-bufferize %s | heir-translate --emit-poulpy | FileCheck %s

// Tensor-form poulpy IR, bufferized and then emitted.

!module = !poulpy.module
!scratch = !poulpy.scratch
!ct = tensor<!poulpy.ciphertext>
!pt = tensor<!poulpy.plaintext>
!sk = !poulpy.secret_key
!tsk = !poulpy.tensor_key
!akm = !poulpy.automorphism_key_map

// CHECK: pub fn chain<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[tsk:v[0-9]+]]: &Tsk<BE>
// CHECK: [[akm:v[0-9]+]]: &Akm<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
// CHECK-NEXT: let mut [[sum:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
// CHECK-NEXT: [[m]].ckks_add_into(&mut [[sum]], &*[[a]], &*[[b]], &mut [[s]].borrow())?;
// CHECK-NEXT: [[m]].ckks_mul_assign(&mut [[sum]], &*[[b]], &*[[tsk]], &mut [[s]].borrow())?;
// CHECK-NEXT: let mut [[rot:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[sum]].base2k(), [[sum]].max_k());
// CHECK-NEXT: [[m]].ckks_rotate_into(&mut [[rot]], &[[sum]], 1i64, &*[[akm]], &mut [[s]].borrow())?;
// CHECK-NEXT: Ok([[rot]])
func.func @chain(%m: !module, %s: !scratch, %tsk: !tsk, %akm: !akm, %a: !ct, %b: !ct) -> !ct {
  %e0 = tensor.empty() : !ct
  %sum = poulpy.add %m, %e0, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> !ct
  %prod = poulpy.mul_assign %m, %sum, %b, %tsk, %s : (!module, !ct, !ct, !tsk, !scratch) -> !ct
  %e1 = tensor.empty() : !ct
  %rot = poulpy.rotate %m, %e1, %prod, %akm, %s <k = 1> : (!module, !ct, !ct, !akm, !scratch) -> !ct
  return %rot : !ct
}

// CHECK: pub fn assign_on_live_arg<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
// CHECK-NEXT: let mut [[copy:v[0-9]+]] = [[a]].clone();
// CHECK-NEXT: [[m]].ckks_add_assign(&mut [[copy]], &*[[b]], &mut [[s]].borrow())?;
// CHECK-NEXT: Ok([[copy]])
func.func @assign_on_live_arg(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  %r = poulpy.add_assign %m, %a, %b, %s : (!module, !ct, !ct, !scratch) -> !ct
  return %r, %a : !ct, !ct
}

// CHECK: pub fn update_arg<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &mut Ct<BE>
// CHECK: [[b:v[0-9]+]]: &Ct<BE>
// CHECK-NEXT: ) -> Result<()> {
// CHECK-NEXT: [[m]].ckks_add_assign(&mut *[[a]], &*[[b]], &mut [[s]].borrow())?;
// CHECK-NEXT: Ok(())
func.func @update_arg(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> !ct {
  %r = poulpy.add_assign %m, %a, %b, %s : (!module, !ct, !ct, !scratch) -> !ct
  return %r : !ct
}

// CHECK: pub fn roundtrip<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[sk:v[0-9]+]]: &Sk<BE>
// CHECK: [[re:v[0-9]+]]: &[f64]
// CHECK: [[im:v[0-9]+]]: &[f64]
// CHECK-NEXT: ) -> Result<(Vec<f64>, Vec<f64>)> {
// CHECK: [[m]].ckks_encode_reim_into(&mut [[pt:v[0-9]+]], &*[[re]], &*[[im]], &mut [[s]].borrow())?;
// CHECK: [[m]].ckks_encrypt_sk(&mut [[ct:v[0-9]+]], &[[pt]], &*[[sk]],
// CHECK: [[m]].ckks_decrypt(&mut [[pt2:v[0-9]+]], &[[ct]], &*[[sk]], &mut [[s]].borrow())?;
// CHECK-NEXT: let mut [[re_out:v[0-9]+]] = vec![0f64; 4];
// CHECK-NEXT: let mut [[im_out:v[0-9]+]] = vec![0f64; 4];
// CHECK-NEXT: [[m]].ckks_decode_reim_into(&[[pt2]], &mut [[re_out]], &mut [[im_out]], &mut [[s]].borrow())?;
// CHECK-NEXT: Ok(([[re_out]], [[im_out]]))
func.func @roundtrip(%m: !module, %s: !scratch, %sk: !sk, %re: tensor<4xf64>, %im: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %pt_empty = tensor.empty() : !pt
  %pt = poulpy.encode %m, %pt_empty, %re, %im, %s <logDelta = 45, logBudget = 20, base2k = 52> : (!module, !pt, tensor<4xf64>, tensor<4xf64>, !scratch) -> !pt
  %ct_empty = tensor.empty() : !ct
  %ct = poulpy.encrypt %m, %ct_empty, %pt, %sk, %s <base2k = 52, ctk = 300> : (!module, !ct, !pt, !sk, !scratch) -> !ct
  %pt2_empty = tensor.empty() : !pt
  %pt2 = poulpy.decrypt %m, %pt2_empty, %ct, %sk, %s : (!module, !pt, !ct, !sk, !scratch) -> !pt
  %re_empty = tensor.empty() : tensor<4xf64>
  %im_empty = tensor.empty() : tensor<4xf64>
  %re_out, %im_out = poulpy.decode %m, %re_empty, %im_empty, %pt2, %s : (!module, tensor<4xf64>, tensor<4xf64>, !pt, !scratch) -> (tensor<4xf64>, tensor<4xf64>)
  return %re_out, %im_out : tensor<4xf64>, tensor<4xf64>
}

// CHECK: pub fn plain_then_neg<BE: PoulpyBackend>(
// CHECK: [[m:v[0-9]+]]: &Module<BE>
// CHECK: [[s:v[0-9]+]]: &mut ScratchOwned<BE>
// CHECK: [[a:v[0-9]+]]: &Ct<BE>
// CHECK: [[pt:v[0-9]+]]: &Pt<BE>
// CHECK-NEXT: ) -> Result<Ct<BE>> {
// CHECK-NEXT: let mut [[sum:v[0-9]+]] = [[m]].ckks_ciphertext_alloc([[a]].base2k(), [[a]].max_k());
// CHECK-NEXT: [[m]].ckks_add_pt_vec_into(&mut [[sum]], &*[[a]], &*[[pt]], &mut [[s]].borrow())?;
// CHECK-NEXT: [[m]].ckks_neg_assign(&mut [[sum]])?;
// CHECK-NEXT: Ok([[sum]])
func.func @plain_then_neg(%m: !module, %s: !scratch, %a: !ct, %pt: !pt) -> !ct {
  %empty = tensor.empty() : !ct
  %sum = poulpy.add_pt_vec %m, %empty, %a, %pt, %s : (!module, !ct, !ct, !pt, !scratch) -> !ct
  %negated = poulpy.neg_assign %m, %sum, %s : (!module, !ct, !scratch) -> !ct
  return %negated : !ct
}
