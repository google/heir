// RUN: heir-translate %s --emit-tfhe-rust --use-levels=False | FileCheck %s

// CHECK: fn load_resource<T: Copy>(path: &str, size: usize) -> Vec<T> {
// CHECK:   use std::fs::File;
// CHECK:   use std::io::Read;
// CHECK:   let mut file = File::open(path).expect("failed to open file");
// CHECK:   let mut buffer = Vec::new();
// CHECK:   file.read_to_end(&mut buffer).expect("failed to read file");
// CHECK:   let expected_bytes = size * std::mem::size_of::<T>();
// CHECK:   assert_eq!(buffer.len(), expected_bytes, "Resource size mismatch");
// CHECK:   let mut data = Vec::with_capacity(size);
// CHECK:   unsafe {
// CHECK:     std::ptr::copy_nonoverlapping(
// CHECK:       buffer.as_ptr(),
// CHECK:       data.as_mut_ptr() as *mut u8,
// CHECK:       expected_bytes,
// CHECK:     );
// CHECK:     data.set_len(size);
// CHECK:   }
// CHECK:   data
// CHECK: }

// CHECK: pub fn test_external_constant(
// CHECK-NEXT:   [[arg0:v[0-9]+]]: &[i32; 4],
// CHECK-NEXT: ) -> [i32; 4] {
// CHECK:   let [[idx:v[0-9]+]] = 0;
// CHECK:   static DATA_[[v0:v[0-9]+]]: OnceLock<Vec<i32>> = OnceLock::new();
// CHECK:   let [[v0]] = DATA_[[v0]].get_or_init(|| load_resource::<i32>("some/path/constant_1.bin", 4));
// CHECK:   let mut [[res:v[0-9]+]] : HashMap<usize, i32> = HashMap::new();
// CHECK:   let [[val0:v[0-9]+]] = [[arg0]][[[idx]]];
// CHECK:   let [[val1:v[0-9]+]] = [[v0]][{{.*}}];
// CHECK:   let [[and:v[0-9]+]] = [[val0]] & [[val1]];
// CHECK:   [[res]].insert([[idx]] as usize, [[and]]);

// CHECK: pub fn test_external_constant_i1(
// CHECK-NEXT:   [[arg0:v[0-9]+]]: &[bool; 4],
// CHECK-NEXT: ) -> [bool; 4] {
// CHECK:   let [[idx:v[0-9]+]] = 0;
// CHECK:   static DATA_[[v0:v[0-9]+]]: OnceLock<Vec<bool>> = OnceLock::new();
// CHECK:   let [[v0]] = DATA_[[v0]].get_or_init(|| load_resource::<bool>("some/path/constant_i1.bin", 4));
// CHECK:   let mut [[res:v[0-9]+]] : HashMap<usize, bool> = HashMap::new();
// CHECK:   let [[val0:v[0-9]+]] = [[arg0]][[[idx]]];
// CHECK:   let [[val1:v[0-9]+]] = [[v0]][{{.*}}];
// CHECK:   let [[and:v[0-9]+]] = [[val0]] & [[val1]];
// CHECK:   [[res]].insert([[idx]] as usize, [[and]]);

module {
  func.func @test_external_constant(%arg0: memref<4xi32>) -> memref<4xi32> {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<4xi32>
    preprocessing.load_resource "some/path/constant_1.bin" into %0
        : (memref<4xi32>) -> ()
    %res = memref.alloc() : memref<4xi32>
    %val0 = memref.load %arg0[%c0] : memref<4xi32>
    %val1 = memref.load %0[%c0] : memref<4xi32>
    %add = arith.andi %val0, %val1 : i32
    memref.store %add, %res[%c0] : memref<4xi32>
    return %res : memref<4xi32>
  }

  func.func @test_external_constant_i1(%arg0: memref<4xi1>) -> memref<4xi1> {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<4xi1>
    preprocessing.load_resource "some/path/constant_i1.bin" into %0
        : (memref<4xi1>) -> ()
    %res = memref.alloc() : memref<4xi1>
    %val0 = memref.load %arg0[%c0] : memref<4xi1>
    %val1 = memref.load %0[%c0] : memref<4xi1>
    %add = arith.andi %val0, %val1 : i1
    memref.store %add, %res[%c0] : memref<4xi1>
    return %res : memref<4xi1>
  }
}
