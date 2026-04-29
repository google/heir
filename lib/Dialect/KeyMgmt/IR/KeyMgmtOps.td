#ifndef LIB_DIALECT_KEYMGMT_IR_KEYMGMTOPS_TD_
#define LIB_DIALECT_KEYMGMT_IR_KEYMGMTOPS_TD_

include "lib/Dialect/KeyMgmt/IR/KeyMgmtDialect.td"
include "lib/Dialect/KeyMgmt/IR/KeyMgmtTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def KeyMgmt_PrefetchKeyOp : Op<KeyMgmt_Dialect, "prefetch_key", [
    MemoryEffects<[MemWrite]>  // Has side effects - modifies cache/memory state
  ]> {
  let summary = "Prefetch a rotation key by index";
  let description = [{
    The prefetch_key operation initiates prefetching of a rotation key
    identified by the given index. This operation has side effects as it
    may trigger memory operations or cache warming.

    This operation should typically be followed by a load_key operation
    with the same index.
  }];

  let arguments = (ins I64:$index);
  let results = (outs);
  let assemblyFormat = "$index attr-dict";
}

def KeyMgmt_LoadKeyOp : Op<KeyMgmt_Dialect, "load_key", [
    MemoryEffects<[MemRead]>  // Reads from memory/cache
  ]> {
  let summary = "Load a rotation key by index";
  let description = [{
    The load_key operation loads a rotation key identified by the given index
    and returns it as a rot_key type.

    The index operand can be:
    - A constant value (e.g., from arith.constant) → produces static rot_key type
    - An affine induction variable or computed index → produces dynamic rot_key type

    The result type indicates whether this is a static or dynamic rotation:
    - !key_mgmt.rot_key<rotation_index = N> for static constant N
    - !key_mgmt.rot_key for dynamic (runtime-determined) indices

    For dynamic rotation keys, analysis passes examine the index operand's
    def-use chain to determine possible values (e.g., from affine loop bounds).

    This operation should typically be preceded by a prefetch_key operation
    with the same index, and followed by a clear_key operation on the result.

    Examples:
      // Static constant index
      %c5 = arith.constant 5 : i64
      %rk = key_mgmt.load_key %c5 : !key_mgmt.rot_key<rotation_index = 5>

      // Dynamic index from affine loop
      affine.for %iv = 1 to 16 {
        %iv_i64 = arith.index_cast %iv : index to i64
        %rk = key_mgmt.load_key %iv_i64 : !key_mgmt.rot_key
      }

      // Can also accept index type directly
      affine.for %iv = 1 to 16 {
        %rk = key_mgmt.load_key %iv : !key_mgmt.rot_key
      }
  }];

  let arguments = (ins AnyTypeOf<[Index, I64]>:$index);
  let results = (outs KeyMgmt_RotKeyType:$rot_key);
  let assemblyFormat = "$index attr-dict `:` type($index) `->` type($rot_key)";
}

def KeyMgmt_UseKeyOp : Op<KeyMgmt_Dialect, "use_key", [Pure]> {
  let summary = "Reference an already-loaded rotation key";
  let description = [{
    The use_key operation converts a static rotation key to a dynamic rotation key
    for use in contexts where the type system requires dynamic keys (e.g., inside loops
    with dynamic indices). This is used for key reuse optimization where a key loaded
    outside a loop can be referenced inside the loop.

    Unlike load_key, this operation does NOT perform any memory operations - it
    simply provides a type conversion from static to dynamic rot_key. The key must
    already be loaded and live.

    The input is typically a static rotation key (!key_mgmt.rot_key<rotation_index = N>)
    and the output is always a dynamic rotation key (!key_mgmt.rot_key<>).

    Example:
      // Load key outside loop
      %c4 = arith.constant 4 : i64
      %rk_static = key_mgmt.load_key %c4 : i64 -> !key_mgmt.rot_key<rotation_index = 4>

      affine.for %iv = 1 to 16 {
        %iv_i64 = arith.index_cast %iv : index to i64
        // When iv == 4, reuse the pre-loaded key instead of loading again
        %rk = scf.if (%iv == 4) {
          %reused = key_mgmt.use_key %rk_static : !key_mgmt.rot_key<rotation_index = 4> -> !key_mgmt.rot_key
          scf.yield %reused
        } else {
          %loaded = key_mgmt.load_key %iv_i64 : i64 -> !key_mgmt.rot_key
          scf.yield %loaded
        }
        ...
      }
  }];

  let arguments = (ins KeyMgmt_RotKeyType:$input_key);
  let results = (outs KeyMgmt_RotKeyType:$rot_key);
  let assemblyFormat = "$input_key attr-dict `:` type($input_key) `->` type($rot_key)";
}

def KeyMgmt_AssumeLoadedOp : Op<KeyMgmt_Dialect, "assume_loaded", [Pure]> {
  let summary = "Assume a rotation key is already loaded in memory";
  let description = [{
    The assume_loaded operation creates a reference to a rotation key that is
    assumed to already be in memory. This is used for post-loop optimization where
    a key loaded inside a loop has its clear operation skipped, leaving it in memory
    for use after the loop.

    Unlike load_key, this operation does NOT perform any I/O or deserialization - it
    simply creates an SSA value representing the key that is already in memory.
    The compiler/optimizer is responsible for ensuring the key is actually loaded.

    This operation is typically emitted as a no-op or debug assertion at runtime.

    Example:
      affine.for %iv = 1 to 16 {
        %rk = key_mgmt.load_key %iv : i64 -> !key_mgmt.rot_key<>
        // use %rk...
        affine.if (%iv != 5) {
          key_mgmt.clear_key %rk : <>  // Skip clearing when iv == 5
        }
      }
      // Key 5 is still in memory (wasn't cleared)
      %c5 = arith.constant 5 : i64
      %rk5 = key_mgmt.assume_loaded %c5 : i64 -> !key_mgmt.rot_key<rotation_index = 5>
      // use %rk5...
      key_mgmt.clear_key %rk5 : <>
  }];

  let arguments = (ins AnyTypeOf<[Index, I64]>:$index);
  let results = (outs KeyMgmt_RotKeyType:$rot_key);
  let assemblyFormat = "$index attr-dict `:` type($index) `->` type($rot_key)";
}

def KeyMgmt_ClearKeyOp : Op<KeyMgmt_Dialect, "clear_key", [
    MemoryEffects<[MemWrite]>  // Has side effects - modifies memory/cache state
  ]> {
  let summary = "Clear a rotation key from memory";
  let description = [{
    The clear_key operation removes a rotation key from memory or cache.
    This operation consumes the rotation key and has side effects.

    This operation should typically be preceded by a load_key operation
    that produced the rotation key being cleared.
  }];

  let arguments = (ins KeyMgmt_RotKeyType:$rot_key);
  let results = (outs);
  let assemblyFormat = "$rot_key attr-dict `:` type($rot_key)";
}



#endif  // LIB_DIALECT_KEYMGMT_IR_KEYMGMTOPS_TD_
