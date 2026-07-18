#ifndef LIB_DIALECT_POULPY_IR_POULPYOPS_TD_
#define LIB_DIALECT_POULPY_IR_POULPYOPS_TD_

include "PoulpyDialect.td"
include "PoulpyTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Poulpy_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poulpy_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def Poulpy_CiphertextMemRef : MemRefOf<[Poulpy_CiphertextType]>;
def Poulpy_UnnormalizedCiphertextMemRef : MemRefOf<[Poulpy_UnnormalizedCiphertextType]>;
def Poulpy_PlaintextMemRef : MemRefOf<[Poulpy_PlaintextType]>;

// maps to ModuleNew::<BE>::new(n)
def Poulpy_ModuleCreateOp : Poulpy_Op<"module_create", [Pure]> {
  let summary = "Create a new poulpy module";
  let description = [{
    See https://docs.rs/poulpy-hal/latest/poulpy_hal/layouts/struct.Module.html
  }];
  let arguments = (ins I64Attr:$N);
  let results = (outs Poulpy_ModuleType:$module);
}

// maps to ScratchOwned::<BE>::alloc(size)
def Poulpy_ScratchAllocOp : Poulpy_Op<"scratch_alloc", [Pure]> {
  let summary = "Allocate a scratch arena of the given byte size";
  let description = [{
    See https://docs.rs/poulpy-hal/latest/poulpy_hal/layouts/struct.ScratchOwned.html
  }];
  let arguments = (ins I64Attr:$size);
  let results = (outs Poulpy_ScratchType:$scratch);
}

// maps to host_module.ckks_pt_vec_alloc + pt.set_meta + encoder.encode_reim
def Poulpy_EncodeOp : Poulpy_Op<"encode"> {
  let summary = "Encode real and imaginary slot vectors into a plaintext";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/encoding/reim/struct.Encoder.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_PlaintextMemRef:$plaintext,
    AnyMemRef:$real,
    AnyMemRef:$imag,
    I64Attr:$logDelta,
    I64Attr:$logBudget
  );
}

// maps to encoder.decode_reim
def Poulpy_DecodeOp : Poulpy_Op<"decode"> {
  let summary = "Decode a plaintext into real and imaginary slot vectors";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/encoding/reim/struct.Encoder.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    AnyMemRef:$real,
    AnyMemRef:$imag,
    Poulpy_PlaintextMemRef:$plaintext
  );
}

// maps to ckks_encrypt_sk
def Poulpy_EncryptOp : Poulpy_Op<"encrypt"> {
  let summary = "Secret-key encryption of a CKKS plaintext";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSEncrypt.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$ciphertext,
    Poulpy_PlaintextMemRef:$plaintext,
    Poulpy_SecretKeyType:$secret_key,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_decrypt
def Poulpy_DecryptOp : Poulpy_Op<"decrypt"> {
  let summary = "Secret-key decryption of a CKKS ciphertext";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSDecrypt.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_PlaintextMemRef:$plaintext,
    Poulpy_CiphertextMemRef:$ciphertext,
    Poulpy_SecretKeyType:$secret_key,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_add_into
def Poulpy_AddOp : Poulpy_Op<"add"> {
  let summary = "Computes `dst = a + b`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_CiphertextMemRef:$b,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_add_assign
def Poulpy_AddAssignOp : Poulpy_Op<"add_assign"> {
  let summary = "Computes `dst += a`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_add_into_unnormalized
def Poulpy_AddUnnormalizedOp : Poulpy_Op<"add_unnormalized"> {
  let summary = "Computes `dst = a + b` without normalizing `dst`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_UnnormalizedCiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_CiphertextMemRef:$b,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_sub_into
def Poulpy_SubOp : Poulpy_Op<"sub"> {
  let summary = "Computes `dst = a - b`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_CiphertextMemRef:$b,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_sub_assign
def Poulpy_SubAssignOp : Poulpy_Op<"sub_assign"> {
  let summary = "Computes `dst -= a`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_sub_into_unnormalized
def Poulpy_SubUnnormalizedOp : Poulpy_Op<"sub_unnormalized"> {
  let summary = "Computes `dst = a - b` without normalizing `dst`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_UnnormalizedCiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_CiphertextMemRef:$b,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_mul_into
def Poulpy_MulOp : Poulpy_Op<"mul"> {
  let summary = "Computes `dst = a * b` using tensor-product keyswitching via `tsk`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSMulOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_CiphertextMemRef:$b,
    Poulpy_TensorKeyType:$tsk,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_mul_assign
def Poulpy_MulAssignOp : Poulpy_Op<"mul_assign"> {
  let summary = "Computes `dst *= a` in-place using tensor-product keyswitching via `tsk`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSMulOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$a,
    Poulpy_TensorKeyType:$tsk,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_rotate_into
def Poulpy_RotateOp : Poulpy_Op<"rotate"> {
  let summary = "Computes `dst = rotate(src, k)`: shifts all complex slots by `k` positions.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSRotateOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$src,
    I64Attr:$k,
    Poulpy_AutomorphismKeyMapType:$keys,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_rotate_assign
def Poulpy_RotateAssignOp : Poulpy_Op<"rotate_assign"> {
  let summary = "Computes `dst = rotate(dst, k)` in-place. Metadata is unchanged.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSRotateOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    I64Attr:$k,
    Poulpy_AutomorphismKeyMapType:$keys,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_div_pow2_into
def Poulpy_RescaleOp : Poulpy_Op<"rescale"> {
  let summary = "Computes `dst = src / 2^bits`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSPow2Ops.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$src,
    I64Attr:$bits,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_div_pow2_assign
def Poulpy_RescaleAssignOp : Poulpy_Op<"rescale_assign"> {
  let summary = "Computes `dst /= 2^bits` in-place.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSPow2Ops.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    I64Attr:$bits,
    Poulpy_ScratchType:$scratch
  );
}

// maps to glwe_normalize
def Poulpy_NormalizeOp : Poulpy_Op<"normalize"> {
  let summary = "Propagate carries in an unnormalized ciphertext, producing a normalized result.";
  let description = [{
    See https://docs.rs/poulpy-core/latest/poulpy_core/api/trait.GLWENormalize.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$res,
    Poulpy_UnnormalizedCiphertextMemRef:$a,
    Poulpy_ScratchType:$scratch
  );
}

// maps to ckks_copy
def Poulpy_CompactLimbsOp : Poulpy_Op<"compact_limbs"> {
  let summary = "Level-aware ciphertext copy that compacts src into a smaller dst.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSCopyOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CiphertextMemRef:$dst,
    Poulpy_CiphertextMemRef:$src,
    Poulpy_ScratchType:$scratch
  );
}

#endif // LIB_DIALECT_POULPY_IR_POULPYOPS_TD_
