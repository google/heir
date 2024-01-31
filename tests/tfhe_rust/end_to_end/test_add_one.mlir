// RUN: heir-opt --cggi-to-tfhe-rust --canonicalize --cse %s | heir-translate --emit-tfhe-rust > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_add_one -- 2 --message_bits=3 | FileCheck %s

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

// CHECK: 00000011
module {
  func.func @fn_under_test(%arg0: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
    %true = arith.constant true
    %false = arith.constant false
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
    %1 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %2 = lwe.trivial_encrypt %1 : !pt_ty to !ct_ty
    %3 = lwe.encode %true {encoding = #encoding} : i1 to !pt_ty
    %4 = lwe.trivial_encrypt %3 : !pt_ty to !ct_ty
    %5 = cggi.lut3(%2, %4, %0) {lookup_table = 8 : ui8} : !ct_ty
    %6 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
    %7 = cggi.lut3(%2, %6, %5) {lookup_table = 150 : ui8} : !ct_ty
    %8 = cggi.lut3(%2, %6, %5) {lookup_table = 23 : ui8} : !ct_ty
    %9 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
    %10 = cggi.lut3(%2, %9, %8) {lookup_table = 43 : ui8} : !ct_ty
    %11 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
    %12 = cggi.lut3(%2, %11, %10) {lookup_table = 43 : ui8} : !ct_ty
    %13 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
    %14 = cggi.lut3(%2, %13, %12) {lookup_table = 43 : ui8} : !ct_ty
    %15 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
    %16 = cggi.lut3(%2, %15, %14) {lookup_table = 43 : ui8} : !ct_ty
    %17 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
    %18 = cggi.lut3(%2, %17, %16) {lookup_table = 105 : ui8} : !ct_ty
    %19 = cggi.lut3(%2, %17, %16) {lookup_table = 43 : ui8} : !ct_ty
    %20 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>
    %21 = cggi.lut3(%2, %20, %19) {lookup_table = 105 : ui8} : !ct_ty
    %22 = cggi.lut3(%2, %4, %0) {lookup_table = 6 : ui8} : !ct_ty
    %23 = cggi.lut3(%2, %9, %8) {lookup_table = 105 : ui8} : !ct_ty
    %24 = cggi.lut3(%2, %11, %10) {lookup_table = 105 : ui8} : !ct_ty
    %25 = cggi.lut3(%2, %13, %12) {lookup_table = 105 : ui8} : !ct_ty
    %26 = cggi.lut3(%2, %15, %14) {lookup_table = 105 : ui8} : !ct_ty
    %from_elements = tensor.from_elements %22, %7, %23, %24, %25, %26, %18, %21 : tensor<8x!ct_ty>
    return %from_elements : tensor<8x!ct_ty>
  }
}
