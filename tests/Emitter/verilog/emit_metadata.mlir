// RUN: heir-translate --allow-unregistered-dialect --emit-metadata %s | FileCheck %s

module {
  func.func @main(%arg0: memref<1x80xi8>) -> memref<1x3x2x1xi8> {
    %c-128_i8 = arith.constant -128 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 2 {
            affine.store %c-128_i8, %alloc[%c0, %arg2, %arg3, %c0] : memref<1x3x2x1xi8>
        }
      }

    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg1 = 0 to 3 {
        affine.for %arg2 = 0 to 2 {
            %12 = affine.load %alloc[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
            affine.store %12, %alloc_0[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
        }
      }
    return %alloc_0 : memref<1x3x2x1xi8>
  }

  func.func @main2(%arg0: memref<80xi8>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : i8
    memref.store %c8, %arg0[%c0] : memref<80xi8>
    return
  }
}

// CHECK:      {
// CHECK-NEXT:   "functions": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "main",
// CHECK-NEXT:       "params": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "index": 0,
// CHECK-NEXT:           "type": {
// CHECK-NEXT:             "memref": {
// CHECK-NEXT:               "element_type": {
// CHECK-NEXT:                 "integer": {
// CHECK-NEXT:                   "is_signed": false,
// CHECK-NEXT:                   "width": 8
// CHECK-NEXT:                 }
// CHECK-NEXT:               },
// CHECK-NEXT:               "shape": [
// CHECK-NEXT:                  1,
// CHECK-NEXT:                  80
// CHECK-NEXT:               ]
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "return_types": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "memref": {
// CHECK-NEXT:             "element_type": {
// CHECK-NEXT:               "integer": {
// CHECK-NEXT:                 "is_signed": false,
// CHECK-NEXT:                 "width": 8
// CHECK-NEXT:               }
// CHECK-NEXT:             },
// CHECK-NEXT:             "shape": [
// CHECK-NEXT:                1,
// CHECK-NEXT:                3,
// CHECK-NEXT:                2,
// CHECK-NEXT:                1
// CHECK-NEXT:             ]
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },

// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "main2",
// CHECK-NEXT:       "params": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "index": 0,
// CHECK-NEXT:           "type": {
// CHECK-NEXT:             "memref": {
// CHECK-NEXT:               "element_type": {
// CHECK-NEXT:                 "integer": {
// CHECK-NEXT:                   "is_signed": false,
// CHECK-NEXT:                   "width": 8
// CHECK-NEXT:                 }
// CHECK-NEXT:               },
// CHECK-NEXT:               "shape": [
// CHECK-NEXT:                  80
// CHECK-NEXT:               ]
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "return_types": []
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
