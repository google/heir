---
title: Data-oblivious Transformations
weight: 9
---

A data-oblivious program is one that decouples data input from program
execution. Such programs exhibit control-flow and memory access patterns that
are independent of their input(s). This programming model, when applied to
encrypted data, is necessary for expressing FHE programs. There are 3 major
transformations applied to convert a conventional program into a data-oblivious
program:

### (1) If-Transformation

If-operations conditioned on inputs create data-dependent control-flow in
programs. `scf.if` operations should at least define a 'then' region (true path)
and always terminate with `scf.yield` even when `scf.if` doesn't produce a
result. To convert a data-dependent `scf.if` operation to an equivalent set of
data-oblivious operations in MLIR, we hoist all safely speculatable operations
in the `scf.if` operation and convert the `scf.yield` operation to an
`arith.select` operation. The following code snippet demonstrates an application
of this transformation.

```mlir
// Before applying If-transformation
func.func @my_function(%input : i1 {secret.secret}) -> () {
  ...
  // Violation: %input is used as a condition causing a data-dependent branch
  %result =`%input -> (i16) {
        %a = arith.muli %b, %c : i16
        scf.yield %a : i16
      } else {
        scf.yield %b : i16
      }
  ...
}

// After applying If-transformation
func.func @my_function(%input : i16 {secret.secret}) -> (){
  ...
  %a = arith.muli %b, %c : i16
  %result = arith.select %input, %a, %b : i16
  ...
}
```

We implement a `ConvertIfToSelect` pass that transforms operations with
secret-input conditions and with only Pure operations (i.e., operations that
have no memory side effect and are speculatable) in their body. **This
transformation cannot be applied to operations when side effects are present in
only one of the two regions.** Although possible, we currently do not support
transformations for operations where both regions have operations with matching
side effects. When side effects are present, the pass fails.

### (2) Loop-Transformation

Loop statements with input-dependent conditions (bounds) and number of
iterations introduce data-dependent branches that violate data-obliviousness. To
convert such loops into a data-oblivious version, we replace input-dependent
conditionals (bounds) with static input-independent parameters (e.g. defining a
constant upper bound), and early-exits with update operations where the value
returned from the loop is selectively updated using conditional predication. In
MLIR, loops are expressed using either `affine.for`, `scf.for` or `scf.while`
operations.

> \[!NOTE\] Early exiting from loops is not supported in `scf` and `affine`, so
> early exits are not supported in this pipeline. Early exits are expected to be
> added to MLIR upstream at some point in the future.

- `affine.for`: This operation lends itself well to expressing data oblivious
  programs because it requires constant loop bounds, eliminating input-dependent
  limits.

```mlir
 %sum_0 = arith.constant 0.0 : f32
 // The for-loop's bound is a fixed constant
 %sum = affine.for %i = 0 to 10 step 2
 iter_args(%sum_iter = %sum_0) -> (f32) {
   %t = affine.load %buffer[%i] : memref<1024xf32>
   %sum_next = arith.addf %sum_iter, %input : f32
   affine.yield %sum_next : f32
 }
 ...
```

- `scf.for`: In contrast to affine.for, scf.for does allow input-dependent
  conditionals which does not adhere to data-obliviousness constraints. A
  solution to this could be to either have the programmer or the compiler
  specify an input-independent upper bound so we can transform the loop to use
  this upper bound and also carefully update values returned from the for-loop
  using conditional predication. Our current solution to this is for the
  programmer to add the lower bound and worst case upper bound in the static
  affine loop's `attributes` list.

```mlir
func.func @my_function(%value: i32 {secret.secret}, %inputIndex: index {secret.secret}) -> i32 {
  ...
  // Violation: for-loop uses %inputIndex as upper bound which causes a secret-dependent control-flow
  %result = scf.for %iv = %begin to %inputIndex step %step_value iter_args(%arg1 = %value) -> i32 {
    %output = arith.muli %arg1, %arg1 : i32
    scf.yield %output : i32
  }{lower = 0, upper = 32}
  ...
}

// After applying Loop-Transformation
func.func @my_function(%value: i32 {secret.secret}, %inputIndex: index {secret.secret}) -> i32 {
  ...
  // Build for-loop using lower and upper values from the `attributes` list
  %result = affine.for %iv = 0 to  step 32 iter_args(%arg1 = %value) -> i32 {
    %output = arith.muli %arg1, %agr1 : i32
    %cond = arith.cmpi eq, %iv, %inputIndex : index
    %newOutput = arith.select %cond, %output, %arg1
    scf.yield %newOutput : i32
    }
  ...
}
```

- `scf.while`: This operation represents a generic while/do-while loop that
  keeps iterating as long as a condition is met. An input-dependent while
  condition introduces a data-dependent control flow that violates
  data-oblivious constraints. For this transformation, the programmer needs to
  add the `max_iter` attribute that describes the maximum number of iterations
  the loop runs which we then use the value to build our static `affine.for`
  loop.

```mlir
// Before applying Loop-Transformation
func.func @my_function(%input: i16 {secret.secret}){
  %zero = arith.constant 0 : i16
  %result = scf.while (%arg1 = %input) : (i16) -> i16 {
    %cond = arith.cmpi slt, %arg1, %zero : i16
    // Violation: scf.while uses %cond whose value depends on %input
    scf.condition(%cond) %arg1 : i16
  } do {
  ^bb0(%arg2: i16):
    %mul = arith.muli %arg2, %arg2: i16
    scf.yield %mul
  } attributes {max_iter = 16 : i64}
  ...
  return
}

// After applying Loop-Transformation
func.func @my_function(%input: i16 {secret.secret}){
  %zero = arith.constant 0 : i16
  %begin = arith.constant 1 : index
  ...
  // Replace while-loop with a for-loop with a constant bound %MAX_ITER
  %result = affine.for %iv = %0 to %16 step %step_value iter_args(%iter_arg = %input) -> i16 {
    %cond = arith.cmpi slt, %iter_arg, %zero : i16
    %mul = arith.muli %iter_arg, %iter_arg : i16
    %output = arith.select %cond, %mul, %iter_arg
    scf.yield %output
  }{max_iter = 16 : i64}
  ...
  return
}

```

### (3) Access-Transformation

Input-dependent memory access cause data-dependent memory footprints. A naive
data-oblivious solution to this maybe doing read-write operations over the
entire data structure while only performing the desired save/update operation
for the index of interest. For simplicity, we only look at load/store operations
for tensors as they are well supported structures in high-level MLIR likely
emitted by most frontends. We drafted the following non-SIMD approach for this
transformation and defer SIMD optimizations to the `heir-simd-vectorizer` pass:

```mlir
// Before applying Access Transformation
func.func @my_function(%input: tensor<16xi32> {secret.secret}, %inputIndex: index {secret.secret}) {
  ...
  %c_10 = arith.constant 10 : i32
  // Violation: tensor.extract loads value at %inputIndex
  %extractedValue = tensor.extract %input[%inputIndex] : tensor<16xi32>
  %newValue = arith.addi %extractedValue, %c_10 : i32
  // Violation: tensor.insert stores value at %inputIndex
  %inserted = tensor.insert %newValue into %input[%inputIndex] : tensor<16xi32>
  ...
}

// After applying Non-SIMD Access Transformation
func.func @my_function(%input: tensor<16xi32> {secret.secret}, %inputIndex: index {secret.secret}) {
  ...
  %c_10 = arith.constant 10 : i32
  %i_0 = arith.constant 0 : index
  %dummyValue = arith.constant 0 : i32

  %extractedValue = affine.for %i=0 to 16 iter_args(%arg= %dummyValue) -> (i32) {
    // 1. Check if %i matches %inputIndex
    // 2. Extract value at %i
    // 3. If %i matches %inputIndex, select %value extracted in (2), else select %dummyValue
    // 4. Yield selected value
    %cond = arith.cmpi eq, %i, %inputIndex : index
    %value = tensor.extract %input[%i] : tensor<16xi32>
    %selected = arith.select %cond, %value, %dummyValue : i32
    affine.yield %selected : i32
  }

  %newValue = arith.addi %extractedValue, %c_10 : i32

  %inserted = affine.for %i=0 to 16 iter_args(%inputArg = %input) -> tensor<16xi32> {
    // 1. Check if %i matches the %inputIndex
    // 2. Insert %newValue and produce %newTensor
    // 3. If %i matches %inputIndex, select %newTensor, else select input tensor
    // 4. Yield final tensor
    %cond = arith.cmpi eq, %i, %inputIndex : index
    %newTensor = tensor.insert %value into %inputArg[%i] : tensor<16xi32>
    %finalTensor= arith.select %cond, %newTensor, %inputArg : tensor<16xi32>
    affine.yield %finalTensor : tensor<16xi32>
  }
  ...
}

```

### More notes on these transformations

These 3 transformations have a cascading behavior where transformations can be
applied progressively to achieve a data-oblivious program. The order of the
transformations goes as follows:

- _Access-Transformation_ (change data-dependent tensor accesses (reads-writes)
  to use `affine.for` and `scf.if` operations) -> _Loop-Transformation_ (change
  data-dependent loops to use constant bounds and condition the loop's yield
  results with `scf.if` operation) -> _If-Transformation_ (substitute
  data-dependent conditionals with `arith.select` operation).
- Besides that, when we apply non-SIMD Access-Transformation on multiple
  data-dependent tensor read-write operations over the same tensor, we can
  benefit from upstream affine transformations over the resulting multiple
  affine loops produced by the Access-Transformation to fuse these loops.
