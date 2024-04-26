#ifndef HEIR_LIB_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
#define HEIR_LIB_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_

include "mlir/Pass/PassBase.td"

def MemrefGlobalReplacePass : Pass<"memref-global-replace", "mlir::ModuleOp"> {
  let summary = "MemrefGlobalReplacePass forwards global memrefs accessors "
  "to arithmetic values";

  let description = [{
    This pass forwards constant global MemRef values to referencing affine
    loads. This pass requires that the MemRef global values are initialized as
    constants and that the affine load access indices are constants (i.e. not
    variadic). Unroll affine loops prior to running this pass.

     MemRef removal is required to remove any memory allocations from the input
     model (for example, TensorFlow models contain global memory holding model
     weights) to support FHE transpilation.

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

  }];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

def ExpandCopyPass : Pass<"expand-copy", "mlir::ModuleOp"> {
  let summary = "Expands memref.copy ops to explicit affine loads and stores";

  let description = [{
    This pass removes memref copy operations by expanding them to affine loads
    and stores. This pass introduces affine loops over the dimensions of the
    MemRef, so must be run prior to any affine loop unrolling in a pipeline.

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

    When `--disable-affine-loop=true` is set, then the output becomes
    ```
    module {
      func.func @memref_copy() {
        %alloc = memref.alloc() : memref<2x3xi32>
        %alloc_0 = memref.alloc() : memref<2x3xi32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %0 = affine.load %alloc[%c0, %c0] : memref<2x3xi32>
        affine.store %0, %alloc_0[%c0, %c0] : memref<2x3xi32>
        %1 = affine.load %alloc[%c0, %c1] : memref<2x3xi32>
        affine.store %1, %alloc_0[%c0, %c1] : memref<2x3xi32>
        %2 = affine.load %alloc[%c0, %c2] : memref<2x3xi32>
        affine.store %2, %alloc_0[%c0, %c2] : memref<2x3xi32>
        [...]
      }
    }
    ```
  }];

  let options = [
    Option<"disableAffineLoop", "disable-affine-loop", "bool", /*default=*/"false",
           "Use this to control to disable using affine loops">,
  ];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

def UnrollAndForwardPass : Pass<"unroll-and-forward", "func::FuncOp"> {
  let summary = "Loop unrolls and forwards stores to loads.";

  let description = [{
    This pass processes the first function in a given module, and, starting from
    the first loop, iteratively does the following:

    1. Fully unroll the loop.
    2. Scan for load ops. For each load op with a statically-inferrable access
    index:
      1. Backtrack to the original memref alloc
      2. Find all store ops at the corresponding index (possibly transitively
      through renames/subviews of the underlying alloc).
      3. Find the last store that occurs and forward it to the load.
      4. If the original memref is an input memref, then forward through any
      renames to make the target load load directly from the argument memref
      (instead of any subviews, say)
    3. Apply the same logic to any remaining loads not inside any for loop.

    This pass requires that tensors are lowered to memref, and only supports
    affine loops with affine.load/store ops.

    Memrefs that result from memref.get_global ops are excluded from
    forwarding, even if they are loaded with a static index, and are instead
    handled by memref-global-replace, which should be run after this pass.

  }];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

def ExtractLoopBodyPass : Pass<"extract-loop-body", "mlir::ModuleOp"> {
  let summary = "Extracts logic of a loop bodies into functions.";

  let description = [{
    This pass extracts logic in the inner body of for loops into functions.

    This pass requires that tensors are lowered to memref. It expects that a
    loop body contains a number of affine.load statements used as inputs to the
    extracted function, and a single affine.store used as the extracted
    function's output.

    Input

    ```
    module {
      func.func @loop_body() {
        %c-128_i8 = arith.constant -128 : i8
        %c127_i8 = arith.constant 127 : i8
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
        affine.for %arg1 = 0 to 25 {
          affine.for %arg2 = 0 to 20 {
            affine.for %arg3 = 0 to 8 {
              %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
              %99 = arith.cmpi slt, %arg0, %c-128_i8 : i8
              %100 = arith.select %99, %c-128_i8, %arg0 : i8
              %101 = arith.cmpi sgt, %arg0, %c127_i8 : i8
              %102 = arith.select %101, %c127_i8, %100 : i8
              affine.store %102, %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
            }
          }
        }
      }
    }
    ```

    Output

    ```
    module {
      func.func @loop_body() {
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
        affine.for %arg1 = 0 to 25 {
          affine.for %arg2 = 0 to 20 {
            affine.for %arg3 = 0 to 8 {
              %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
              %102 = func.call @__for_loop(%98) : (i8) -> i8
              affine.store %102, %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
            }
          }
        }
      }
      func.func private @__for_loop(%arg0: i8) -> i8 {
        %c-128_i8 = arith.constant -128 : i8
        %c127_i8 = arith.constant 127 : i8
        %99 = arith.cmpi slt, %arg0, %c-128_i8 : i8
        %100 = arith.select %99, %c-128_i8, %arg0 : i8
        %101 = arith.cmpi sgt, %arg0, %c127_i8 : i8
        %102 = arith.select %101, %c127_i8, %100 : i8
        return %102 : i8
      }
    }
    ```
  }];

  let options = [
    Option<"minimumLoopSize", "min-loop-size", "unsigned", /*default=*/"4",
           "Use this to control the minimum loop size to apply this pass">,
    Option<"minimumBodySize", "min-body-size", "unsigned", /*default=*/"4",
           "Use this to control the minimum loop body size to apply this pass">,
  ];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // HEIR_LIB_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_TD_
