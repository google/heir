// This tests functional correctness through heir-translate for memref lookups.
//
// RUN: run_verilog \
// RUN:  --verilog_module %s \
// RUN:  --input='arg1=1;arg2=0' \
// RUN:  --input='arg1=1;arg2=1' \
// RUN:  --input='arg1=1;arg2=5' \
// RUN:  --input='arg1=1;arg2=9' \
// RUN:  > %t
// RUN: FileCheck %s < %t

// CHECK: b'\x00\x0b'
// CHECK-NEXT: b'\x00\x16'
// CHECK-NEXT: b'\x00\x42'
// CHECK-NEXT: b'\x00\x6f'

module main(
  input wire arg1,
  input wire signed [4:0] arg2,
  output wire signed [15:0] _out_
);
  wire signed [159:0] v3;
  wire signed [159:0] v4;
  wire signed [15:0] v5;
  wire signed [15:0] v6;
  wire signed [15:0] v7;
  assign v3 = 160'h006F00630058004D00420037002C00210016000B;
  assign v4 = 160'h0069005F0055004B00410037002D00230019000F;

  assign v5 = v3[15 + 16 * arg2 : 16 * arg2];
  assign v6 = v4[15 + 16 * arg2 : 16 * arg2];
  assign v7 = arg1 ? v5 : v6;
  assign _out_ = v7;
endmodule

// The verilog is translated from the following program:
//
// module {
//   memref.global "private" constant @__constant_10xi16 : memref<10xi16> = dense<[11, 22, 33, 44, 55, 66, 77, 88, 99, 111]>
//   memref.global "private" constant @__constant_10xi16_hex : memref<10xi16> = dense<"0x0F00190023002D00370041004B0055005F006900">
//   // The first argument selects the memref, the second indexes the memref.
//   func.func @main(%arg0: i1, %arg1: i5) -> i16 {
//     %1 = memref.get_global @__constant_10xi16 : memref<10xi16>
//     %2 = memref.get_global @__constant_10xi16_hex : memref<10xi16>
//     %3 = arith.index_cast %arg1 : i4 to index
//     %4 = affine.load %1[%3] : memref<10xi16>
//     %5 = affine.load %2[%3] : memref<10xi16>
//     %6 = arith.select %arg0, %4, %5 : i16
//     return %6 : i16
//   }
// }
