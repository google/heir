// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// CHECK: var (
// CHECK:   g_v{{.*}} []int32
// CHECK:   g_v{{.*}} []bool
// CHECK: )

// CHECK: func init() {
// CHECK:   var err error
// CHECK:   g_v{{.*}}, err = loadResource_int32("some/path/constant_1.bin", 4)
// CHECK:   g_v{{.*}}, err = loadResource_bool("some/path/constant_i1.bin", 4)
// CHECK:   if err != nil {
// CHECK:     panic(err)
// CHECK:   }
// CHECK: }

// CHECK: func loadResource_bool(path string, size int) ([]bool, error) {
// CHECK:   file, err := os.Open(path)
// CHECK:   if err != nil {
// CHECK:     return nil, err
// CHECK:   }
// CHECK:   defer file.Close()
// CHECK:   data := make([]bool, size)
// CHECK:   err = binary.Read(file, binary.LittleEndian, &data)
// CHECK:   if err != nil {
// CHECK:     return nil, err
// CHECK:   }
// CHECK:   return data, nil
// CHECK: }

// CHECK: func loadResource_int32(path string, size int) ([]int32, error) {
// CHECK:   file, err := os.Open(path)
// CHECK:   if err != nil {
// CHECK:     return nil, err
// CHECK:   }
// CHECK:   defer file.Close()
// CHECK:   data := make([]int32, size)
// CHECK:   err = binary.Read(file, binary.LittleEndian, &data)
// CHECK:   if err != nil {
// CHECK:     return nil, err
// CHECK:   }
// CHECK:   return data, nil
// CHECK: }

// CHECK: func Test_external_constant(v{{.*}} []int32) ([]int32) {
// CHECK-NOT: g_v{{.*}} :=
// CHECK:   v{{.*}} := make([]int32, 4)
// CHECK:   for v{{.*}} := 0; v{{.*}} < 4; v{{.*}} += 1 {
// CHECK:     v{{.*}} := v{{.*}}[v{{.*}}]
// CHECK:     v{{.*}} := g_v{{.*}}[v{{.*}}]
// CHECK:     v{{.*}} := v{{.*}} + v{{.*}}
// CHECK:     v{{.*}}[v{{.*}}] = v{{.*}}
// CHECK:   }
// CHECK:   return v{{.*}}
// CHECK: }

// CHECK: func Test_external_constant_i1(v{{.*}} []bool) ([]bool) {
// CHECK-NOT: g_v{{.*}} :=
// CHECK:   v{{.*}} := make([]bool, 4)
// CHECK:   for v{{.*}} := 0; v{{.*}} < 4; v{{.*}} += 1 {
// CHECK:     v{{.*}} := v{{.*}}[v{{.*}}]
// CHECK:     v{{.*}} := g_v{{.*}}[v{{.*}}]
// CHECK:     v{{.*}} := v{{.*}} && v{{.*}}
// CHECK:     v{{.*}}[v{{.*}}] = v{{.*}}
// CHECK:   }
// CHECK:   return v{{.*}}
// CHECK: }

module attributes {scheme.bgv} {
  func.func @test_external_constant(%arg0: memref<4xi32>) -> memref<4xi32> {
    %0 = preprocessing.load_resource "some/path/constant_1.bin" : memref<4xi32>
    %res = memref.alloc() : memref<4xi32>
    affine.for %i = 0 to 4 {
      %val0 = memref.load %arg0[%i] : memref<4xi32>
      %val1 = memref.load %0[%i] : memref<4xi32>
      %add = arith.addi %val0, %val1 : i32
      memref.store %add, %res[%i] : memref<4xi32>
    }
    return %res : memref<4xi32>
  }

  func.func @test_external_constant_i1(%arg0: memref<4xi1>) -> memref<4xi1> {
    %0 = preprocessing.load_resource "some/path/constant_i1.bin" : memref<4xi1>
    %res = memref.alloc() : memref<4xi1>
    affine.for %i = 0 to 4 {
      %val0 = memref.load %arg0[%i] : memref<4xi1>
      %val1 = memref.load %0[%i] : memref<4xi1>
      %and = arith.andi %val0, %val1 : i1
      memref.store %and, %res[%i] : memref<4xi1>
    }
    return %res : memref<4xi1>
  }
}
