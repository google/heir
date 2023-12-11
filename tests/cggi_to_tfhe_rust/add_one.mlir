// RUN: heir-opt --cggi-to-tfhe-rust -cse %s | FileCheck %s

// This function computes add_one to an i8 input split into bits and returns
// the resulting i8 split into bits.
// CHECK-LABEL: add_one
// CHECK-NOT: cggi
// CHECK-NOT: lwe
// CHECK-COUNT-11: tfhe_rust.apply_lookup_table
// CHECK: [[CONCAT:%.*]] = tensor.from_elements
// CHECK: return [[CONCAT]] : tensor<8x!tfhe_rust.eui3>

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

module {
  func.func @add_one(%arg0: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
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
    %extracted = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
    %0 = lwe.encode %true {encoding = #encoding} : i1 to !pt_ty
    %1 = lwe.trivial_encrypt %0 : !pt_ty to !ct_ty
    %2 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %3 = lwe.trivial_encrypt %2 : !pt_ty to !ct_ty
    %4 = cggi.lut3(%extracted, %1, %3) {lookup_table = 8 : ui8} : !ct_ty
    %extracted_0 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
    %5 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %6 = lwe.trivial_encrypt %5 : !pt_ty to !ct_ty
    %7 = cggi.lut3(%4, %extracted_0, %6) {lookup_table = 150 : ui8} : !ct_ty
    %8 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %9 = lwe.trivial_encrypt %8 : !pt_ty to !ct_ty
    %10 = cggi.lut3(%4, %extracted_0, %9) {lookup_table = 23 : ui8} : !ct_ty
    %extracted_1 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
    %11 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %12 = lwe.trivial_encrypt %11 : !pt_ty to !ct_ty
    %13 = cggi.lut3(%10, %extracted_1, %12) {lookup_table = 43 : ui8} : !ct_ty
    %extracted_2 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
    %14 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %15 = lwe.trivial_encrypt %14 : !pt_ty to !ct_ty
    %16 = cggi.lut3(%13, %extracted_2, %15) {lookup_table = 43 : ui8} : !ct_ty
    %extracted_3 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
    %17 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %18 = lwe.trivial_encrypt %17 : !pt_ty to !ct_ty
    %19 = cggi.lut3(%16, %extracted_3, %18) {lookup_table = 43 : ui8} : !ct_ty
    %extracted_4 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
    %20 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %21 = lwe.trivial_encrypt %20 : !pt_ty to !ct_ty
    %22 = cggi.lut3(%19, %extracted_4, %21) {lookup_table = 43 : ui8} : !ct_ty
    %extracted_5 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
    %23 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %24 = lwe.trivial_encrypt %23 : !pt_ty to !ct_ty
    %25 = cggi.lut3(%22, %extracted_5, %24) {lookup_table = 105 : ui8} : !ct_ty
    %26 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %27 = lwe.trivial_encrypt %26 : !pt_ty to !ct_ty
    %28 = cggi.lut3(%22, %extracted_5, %27) {lookup_table = 43 : ui8} : !ct_ty
    %extracted_6 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>
    %29 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %30 = lwe.trivial_encrypt %29 : !pt_ty to !ct_ty
    %31 = cggi.lut3(%28, %extracted_6, %30) {lookup_table = 105 : ui8} : !ct_ty
    %32 = lwe.encode %true {encoding = #encoding} : i1 to !pt_ty
    %33 = lwe.trivial_encrypt %32 : !pt_ty to !ct_ty
    %34 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %35 = lwe.trivial_encrypt %34 : !pt_ty to !ct_ty
    %36 = cggi.lut3(%extracted, %33, %35) {lookup_table = 6 : ui8} : !ct_ty
    %37 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %38 = lwe.trivial_encrypt %37 : !pt_ty to !ct_ty
    %39 = cggi.lut3(%10, %extracted_1, %38) {lookup_table = 105 : ui8} : !ct_ty
    %40 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %41 = lwe.trivial_encrypt %40 : !pt_ty to !ct_ty
    %42 = cggi.lut3(%13, %extracted_2, %41) {lookup_table = 105 : ui8} : !ct_ty
    %43 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %44 = lwe.trivial_encrypt %43 : !pt_ty to !ct_ty
    %45 = cggi.lut3(%16, %extracted_3, %44) {lookup_table = 105 : ui8} : !ct_ty
    %46 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
    %47 = lwe.trivial_encrypt %46 : !pt_ty to !ct_ty
    %48 = cggi.lut3(%19, %extracted_4, %47) {lookup_table = 105 : ui8} : !ct_ty
    %from_elements = tensor.from_elements %31, %25, %48, %45, %42, %39, %7, %36 : tensor<8x!ct_ty>
    return %from_elements : tensor<8x!ct_ty>
  }
}
