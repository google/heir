// RUN: heir-opt --convert-if-to-select --mlir-to-cggi=abc-fast=true --scheme-to-tfhe-rs %s | FileCheck %s

// def mul_gf256_2(x: int, y: int) -> int :
//     '''Rinjdael multiplication in the ring GF(256) with RHS limited to 2 bits'''
//     z = 0
//     for i in range(2):
//         if y & (1 < i):
//             z ^= x
//         x = (x < 1)
//         if x >= 256:
//             x = (x & 255) ^ 27
//     return z

// CHECK: func @main
func.func @main(%x: i8 {secret.secret}, %y: i8) -> (i8) {
    %z = arith.constant 0 : i8
    %2:2 = affine.for %4 = 0 to 2 iter_args(%5 = %x, %6 = %z) -> (i8, i8) {
      %7 = arith.constant 1 : i8
      %44 = arith.index_cast %4 : index to i8
      %8 = arith.shli %7, %44 : i8
      %9 = arith.andi %y, %8 : i8
      %91 = arith.trunci %9 : i8 to i1
      %10 = arith.xori %6, %5 : i8
      %argz = arith.select %91, %10, %6 : i8
      %11 = arith.constant 1 : i8
      %12 = arith.shli %argz, %11 : i8
      %13 = arith.constant 0x80 : i8
      %14 = arith.cmpi sge, %12, %13 : i8
      %argxx = scf.if %14 -> (i8) {
        %15 = arith.constant 255 : i8
        %16 = arith.andi %12, %15 : i8
        %17 = arith.constant 27 : i8
        %122 = arith.xori %16, %17 : i8
        scf.yield %122 : i8
      } else {
        scf.yield %12 : i8
      }
      affine.yield %argxx, %argz : i8, i8
    }
    func.return %2#1 : i8
}
