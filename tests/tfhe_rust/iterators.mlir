// RUN: heir-translate %s --emit-tfhe-rust | FileCheck %s

// CHECK: enum GateInput

// CHECK: enum OpType

module {
  // CHECK-LABEL: pub fn generate_cleartext_ops
  // CHECK-NEXT:   [[sks:v[0-9]+]]: &ServerKey,
  // CHECK-NEXT:   [[input:v[0-9]+]]: &[i16; 2],
  // CHECK-NEXT: ) -> {{[[][[]}}i8; 8]; 2] {
  // CHECK:      let mut [[v1:.*]] : HashMap<(usize, usize), i8> = HashMap::new();
  // CHECK-NEXT: for [[v2:.*]] in 0..2 {
  // CHECK-NEXT:   let [[v3:.*]] = [[input]][[[v2]]];
  // CHECK-NEXT:   for [[v4:.*]] in 0..8 {
  // CHECK-NEXT:     let [[v5:.*]] = [[v4]] as i16;
  // CHECK-NEXT:     let [[v6:.*]] = [[v5]] & [[v3]];
  // CHECK-NEXT:     let [[v7:.*]] = [[v6]] as i8;
  // CHECK-NEXT:     [[v1]].insert(([[v2]] as usize, [[v4]] as usize), [[v7]]);
  // CHECK-NEXT:     }
  // CHECK-NEXT:  }
  // CHECK-NEXT: core::array::from_fn
  // CHECK-SAME:   [[v1]].get(&([[i0:.*]], [[i1:.*]])).unwrap().clone()
  func.func @generate_cleartext_ops(%sks : !tfhe_rust.server_key, %arg0 : memref<2xi16>) -> (memref<2x8xi8>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x8xi8>
    affine.for %arg1 = 0 to 2 {
      %0 = memref.load %arg0[%arg1] : memref<2xi16>
      affine.for %arg2 = 0 to 8 {
        %1 = arith.index_cast %arg2 : index to i16
        %2 = arith.andi %1, %0 : i16
        %3 = arith.trunci %2 : i16 to i8
        memref.store %3, %alloc[%arg1, %arg2] :memref<2x8xi8>
      }
    }
    return %alloc : memref<2x8xi8>
  }

  // A memref is stored with an initial value and then iteratively summed
  // CHECK-LABEL: pub fn iterative
  // CHECK-NEXT:   [[input1:v[0-9]+]]: &{{[[][[]}}i8; 16]; 1],
  // CHECK-NEXT:   [[input2:v[0-9]+]]: &{{[[][[]}}i8; 1]; 16],
  // CHECK-NEXT: ) -> {{[[][[]}}i8; 1]; 1] {
  // CHECK:      let [[v0:.*]] = 29;
  // CHECK-NEXT: let mut [[v1:.*]] : HashMap<(usize, usize), i8> =
  // CHECK-NEXT: for [[v2:.*]] in 0..1 {
  // CHECK-NEXT:   for [[v3:.*]] in 0..1 {
  // CHECK-NEXT:     [[v1]].insert(([[v2]] as usize, [[v3]] as usize), [[v0]]);
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: for [[v4:.*]] in 0..1 {
  // CHECK-NEXT:   for [[v5:.*]] in 0..1 {
  // CHECK-NEXT:     for [[v6:.*]] in 0..16 {
  // CHECK-NEXT:       let [[v7:.*]] = [[input1]][[[v4]]][[[v6]]];
  // CHECK-NEXT:       let [[v8:.*]] = [[input2]][[[v6]]][[[v5]]];
  // CHECK-NEXT:       let [[v9:.*]] = [[v1]].get(&([[v4]] as usize, [[v5]] as usize)).unwrap();
  // CHECK-NEXT:       let [[v10:.*]] = [[v7]] & [[v8]];
  // CHECK-NEXT:       let [[v11:.*]] = [[v10]] & [[v9]];
  // CHECK-NEXT:       [[v1]].insert(([[v4]] as usize, [[v5]] as usize), [[v11]]);
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: let mut [[v12:.*]] : HashMap<(usize, usize), i8> = HashMap::new();
  // CHECK-NEXT: for [[v13:.*]] in 0..1 {
  // CHECK-NEXT:   for [[v14:.*]] in 0..16 {
  // CHECK-NEXT:     let [[v15:.*]] = [[v1]].get(&([[v13]] as usize, [[v14]] as usize)).unwrap();
  // CHECK-NEXT:     let [[v16:.*]] = [[v15]] as i8;
  // CHECK-NEXT:     [[v12]].insert(([[v13]] as usize, [[v14]] as usize), [[v16]]);
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: core::array::from_fn
  // CHECK-SAME:   [[v12]].get(&([[i0:.*]], [[i1:.*]])).unwrap().clone()
  func.func @iterative(%alloc_6: memref<1x16xi8>, %alloc_7 : memref<16x1xi8>) -> memref<1x1xi4> {
    %c29_i8 = arith.constant 29 : i8
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        memref.store %c29_i8, %alloc[%arg1, %arg2] : memref<1x1xi8>
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %5 = memref.load %alloc_6[%arg1, %arg3] : memref<1x16xi8>
          %6 = memref.load %alloc_7[%arg3, %arg2] : memref<16x1xi8>
          %7 = memref.load %alloc[%arg1, %arg2] : memref<1x1xi8>
          %8 = arith.andi %5, %6 : i8
          %13 = arith.andi %8, %7 : i8
          memref.store %13, %alloc[%arg1, %arg2] : memref<1x1xi8>
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi4>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %5 = memref.load %alloc[%arg1, %arg2] : memref<1x1xi8>
        %6 = arith.trunci %5 : i8 to i4
        memref.store %6, %alloc_1[%arg1, %arg2] : memref<1x1xi4>
      }
    }
    return %alloc_1 : memref<1x1xi4>
  }
}
