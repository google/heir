// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// Resource-backed globals (dense_resource<...>) store their element bytes
// out-of-line in the dialect_resources blob. The emitter must read the blob
// and print the elements inline, just like an inline dense<...> attribute.

module attributes {backend.lattigo, scheme.ckks} {

// CHECK: var __constant_1x3xf32 = []float32{-1.697694e-01, -3.044266e-01, -5.597579e-01}
memref.global "private" constant @__constant_1x3xf32 : memref<1x3xf32> = dense_resource<torch_tensor_1_3_torch.float32> {alignment = 64 : i64}

// CHECK: var __constant_3xi8 = []int8{1, -2, 127}
memref.global "private" constant @__constant_3xi8 : memref<3xi8> = dense_resource<blob_i8>

// CHECK: var __constant_2xi16 = []int16{256, -1}
memref.global "private" constant @__constant_2xi16 : memref<2xi16> = dense_resource<blob_i16>

// CHECK: var __constant_2xi64 = []int64{1, -1}
memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense_resource<blob_i64>

// i1 elements are stored one byte each and emitted as Go bool literals.
// CHECK: var __constant_3xi1 = []bool{true, false, true}
memref.global "private" constant @__constant_3xi1 : memref<3xi1> = dense_resource<blob_i1>

// CHECK: func test_global() ([]float32) {
func.func @test_global() -> memref<1x3xf32> {
  // CHECK: v{{.*}} := __constant_1x3xf32
  %global = memref.get_global @__constant_1x3xf32 : memref<1x3xf32>
  return %global : memref<1x3xf32>
}

}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_3_torch.float32: "0x0400000009D82DBECCDD9BBE4B4C0FBF",
      blob_i8: "0x0100000001FE7F",
      blob_i16: "0x020000000001FFFF",
      blob_i64: "0x080000000100000000000000FFFFFFFFFFFFFFFF",
      blob_i1: "0x01000000010001"
    }
  }
#-}
