---
title: Passes
weight: 2
---

## Memref Global Replace

This pass forwards constant global MemRef values to referencing affine loads.
This pass requires that the MemRef global values are initialized as constants.
It also requires that the affine load access indices are constants (i.e. not
variadic or symbolic), so loops must be unrolled prior to this pass.

Input
```
module {
  memref.global "private" constant @__constant_8xi16 : memref<2x4xi16> = dense<[[-10, 20, 3, 4], [5, 6, 7, 8]]>
  func.func @main() -> i16 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.get_global @__constant_8xi16 : memref<2x4xi16>
    %1 = affine.load %0[%c1, %c1 + %c2] : memref<2x4xi16>
    return %1 : i16
  }
}
```

Output
```
module {
  func.func @main() -> i16 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8_i16 = arith.constant 8 : i16
    return %c8_i16 : i16
  }
}
```

## Expand Copy

This pass rewrites memref copy operations by expanding them to affine loads and
stores. This pass introduces affine loops over the dimensions of the memref.

Input

```
module {
  func.func @memref_copy() {
    %alloc = memref.alloc() : memref<2x3xi32>
    %alloc_0 = memref.alloc() : memref<2x3xi32>
    memref.copy %alloc, %alloc_0 : memref<1x1xi32> to memref<1x1xi32>
  }
}
```

Output

```
module {
  func.func @memref_copy() {
    %alloc = memref.alloc() : memref<2x3xi32>
    %alloc_0 = memref.alloc() : memref<2x3xi32>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 3 {
        %1 = affine.load %alloc[%arg0, %arg1] : memref<2x3xi32>
        affine.store %1, %alloc_0[%arg0, %arg1] : memref<2x3xi32>
      }
    }
  }
}
```
