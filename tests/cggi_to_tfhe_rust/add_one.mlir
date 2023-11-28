// RUN: heir-opt --cggi-to-tfhe-rust -cse %s | FileCheck %s

// This function computes add_one to an i8 input split into bits and returns
// the resulting i8 split into bits.
// CHECK-LABEL: add_one
// CHECK-NOT: cggi
// CHECK-NOT: lwe
// CHECK-COUNT-11: tfhe_rust.apply_lookup_table
#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>
func.func @add_one(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty, %arg4: !ct_ty, %arg5: !ct_ty, %arg6: !ct_ty, %arg7: !ct_ty) -> (!ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty) {
  %false = arith.constant false
  %0 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %1 = lwe.trivial_encrypt %0 : !pt_ty to !ct_ty
  %2 = cggi.lut3(%arg0, %arg1, %1) {lookup_table = 6 : ui8} : !ct_ty
  %3 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %4 = lwe.trivial_encrypt %3 : !pt_ty to !ct_ty
  %5 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %6 = lwe.trivial_encrypt %5 : !pt_ty to !ct_ty
  %7 = cggi.lut3(%arg0, %4, %6) {lookup_table = 1 : ui8} : !ct_ty
  %8 = cggi.lut3(%arg0, %arg1, %arg2) {lookup_table = 120 : ui8} : !ct_ty
  %9 = cggi.lut3(%arg0, %arg1, %arg2) {lookup_table = 128 : ui8} : !ct_ty
  %10 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %11 = lwe.trivial_encrypt %10 : !pt_ty to !ct_ty
  %12 = cggi.lut3(%9, %arg3, %11) {lookup_table = 6 : ui8} : !ct_ty
  %13 = cggi.lut3(%9, %arg3, %arg4) {lookup_table = 120 : ui8} : !ct_ty
  %14 = cggi.lut3(%9, %arg3, %arg4) {lookup_table = 128 : ui8} : !ct_ty
  %15 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %16 = lwe.trivial_encrypt %15 : !pt_ty to !ct_ty
  %17 = cggi.lut3(%14, %arg5, %16) {lookup_table = 6 : ui8} : !ct_ty
  %18 = cggi.lut3(%14, %arg5, %arg6) {lookup_table = 120 : ui8} : !ct_ty
  %19 = cggi.lut3(%14, %arg5, %arg6) {lookup_table = 128 : ui8} : !ct_ty
  %20 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %21 = lwe.trivial_encrypt %20 : !pt_ty to !ct_ty
  %22 = cggi.lut3(%19, %arg7, %21) {lookup_table = 6 : ui8} : !ct_ty
  return %22, %18, %17, %13, %12, %8, %2, %7 : !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty, !ct_ty
}
