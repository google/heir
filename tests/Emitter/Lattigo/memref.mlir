// RUN: heir-translate %s --emit-lattigo | FileCheck %s

module attributes {backend.lattigo, scheme.ckks} {

memref.global "private" constant @__constant_1024xf32 : memref<1024xf32> = dense<1.000000e+00>

// CHECK: var __constant_4xi32 = []int32{1, 2, 3, 4}
memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense_resource<weights>

// CHECK: func test_memref(_ []float32) ([]float32, int64) {
func.func @test_memref(%arg0: memref<1024xf32>) -> (memref<1024xf32>, index) {
  %c0 = arith.constant 0 : index

  // CHECK: v{{.*}} := make([]float32, 1024)
  %alloc = memref.alloc() : memref<1024xf32>

  // CHECK: v{{.*}} := __constant_1024xf32
  %global = memref.get_global @__constant_1024xf32 : memref<1024xf32>

  // CHECK: copy(v{{.*}}, v{{.*}})
  memref.copy %global, %alloc : memref<1024xf32> to memref<1024xf32>

  // CHECK: v{{.*}} := v{{.*}}[v{{.*}}]
  %val = memref.load %alloc[%c0] : memref<1024xf32>

  // CHECK: v{{.*}}[v{{.*}}] = v{{.*}}
  memref.store %val, %alloc[%c0] : memref<1024xf32>

  // CHECK: v{{.*}} := v{{.*}}[0 : (0 + 1024)]
  %subview = memref.subview %alloc[0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>

  // CHECK: v{{.*}} := v{{.*}}
  %cast = memref.cast %subview : memref<1024xf32, strided<[1]>> to memref<1024xf32, strided<[?], offset: ?>>



  // CHECK: v{{.*}} := int64(len(v{{.*}}))
  %dim = memref.dim %cast, %c0 : memref<1024xf32, strided<[?], offset: ?>>

  return %alloc, %dim : memref<1024xf32>, index
}

// CHECK: func test_shape(v{{.*}} []float32) ([]float32) {
func.func @test_shape(%arg0: memref<1024xf32>) -> memref<1024xf32> {
  // CHECK: v{{.*}} := v{{.*}}
  %expand = memref.expand_shape %arg0 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>

  // CHECK: v{{.*}} := v{{.*}}
  %collapse = memref.collapse_shape %expand [[0, 1]] : memref<1x1024xf32> into memref<1024xf32>

  return %collapse : memref<1024xf32>
}

}

{-#
  dialect_resources: {
    builtin: {
      weights: "0x4000000001000000020000000300000004000000"
    }
  }
#-}
