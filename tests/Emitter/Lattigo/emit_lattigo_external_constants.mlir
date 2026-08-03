// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// Go requires a package name; --package-name defaults to main.
// CHECK: package main

// CHECK: var (
// Globals live at module scope, so their names must be distinct across
// functions -- value names are only unique within a function.
// CHECK:   g_resource0 []int32
// CHECK:   g_resource1 []bool
// CHECK: )

// CHECK: func init() {
// CHECK:   var err error
// CHECK:   g_resource0, err = loadResource_int32("some/path/constant_1.bin", 4)
// CHECK:   g_resource1, err = loadResource_bool("some/path/constant_i1.bin", 4)
// CHECK:   if err != nil {
// CHECK:     panic(err)
// CHECK:   }
// CHECK: }

// CHECK: func loadResource_bool(path string, size int) ([]bool, error) {
// CHECK:   resolvedPath := heirResolvePath(path)
// CHECK:   file, err := os.Open(resolvedPath)
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
// CHECK:   resolvedPath := heirResolvePath(path)
// CHECK:   file, err := os.Open(resolvedPath)
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

// CHECK: func heirResolvePath(path string) string {
// CHECK:   if srcDir := os.Getenv("TEST_SRCDIR"); srcDir != "" {
// CHECK:     if workspace := os.Getenv("TEST_WORKSPACE"); workspace != "" {
// CHECK:       return filepath.Join(srcDir, workspace, path)
// CHECK:     }
// CHECK:   }
// CHECK:   return path
// CHECK: }

// CHECK: func Test_external_constant(v{{.*}} []int32) ([]int32) {
// CHECK-NOT: g_resource{{[0-9]+}} :=
// CHECK:   v{{.*}} := make([]int32, 4)
// CHECK:   for v{{.*}} := 0; v{{.*}} < 4; v{{.*}} += 1 {
// CHECK:     v{{.*}} := v{{.*}}[v{{.*}}]
// CHECK:     v{{.*}} := g_resource0[v{{.*}}]
// CHECK:     v{{.*}} := v{{.*}} + v{{.*}}
// CHECK:     v{{.*}}[v{{.*}}] = v{{.*}}
// CHECK:   }
// CHECK:   return v{{.*}}
// CHECK: }

// CHECK: func Test_external_constant_i1(v{{.*}} []bool) ([]bool) {
// CHECK-NOT: g_resource{{[0-9]+}} :=
// CHECK:   v{{.*}} := make([]bool, 4)
// CHECK:   for v{{.*}} := 0; v{{.*}} < 4; v{{.*}} += 1 {
// CHECK:     v{{.*}} := v{{.*}}[v{{.*}}]
// CHECK:     v{{.*}} := g_resource1[v{{.*}}]
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
