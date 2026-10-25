#ifndef LIB_DIALECT_POULPY_IR_POULPYOPS_TD_
#define LIB_DIALECT_POULPY_IR_POULPYOPS_TD_

include "PoulpyDialect.td"
include "PoulpyTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Poulpy_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poulpy_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

// Every op that writes a ciphertext, plaintext or slot vector is modeled as a
// destination-passing-style (DPS) op, matching the Rust API
// (`module.ckks_add_into(dst, a, b, scratch)`): the destination operand is the
// DPS init and the result is tied 1:1 to it. The op is valid both in tensor
// form (value semantics, one result per init, before bufferization) and in
// buffer form (memref operands, no results, after one-shot-bufferize). The
// `_assign` ops also read their destination.
class Poulpy_DpsOp<string mnemonic, list<Trait> traits = [],
                   string initName = "Dst", int initCount = 1,
                   bit readsInit = 0> :
        Poulpy_Op<mnemonic, traits # [DestinationStyleOpInterface,
            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), get}] # initName # [{Mutable().getOperandNumber(),
          }] # !cast<string>(initCount) # [{);
    }
    static constexpr bool readsDpsInit() { return }] #
        !if(readsInit, "true", "false") # [{; }
  }];
  // Only memref operands carry effects, so the tensor form stays free of side
  // effects while the buffer form reads its inputs and writes its destination.
  let extraClassDefinition = [{
    void $cppClass::getEffects(::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
            &effects) {
      for (::mlir::OpOperand &operand : getOperation()->getOpOperands()) {
        if (!::llvm::isa<::mlir::MemRefType>(operand.get().getType()))
          continue;
        bool isInit = isDpsInit(&operand);
        if (!isInit || readsDpsInit())
          effects.emplace_back(::mlir::MemoryEffects::Read::get(), &operand,
                               ::mlir::SideEffects::DefaultResource::get());
        if (isInit)
          effects.emplace_back(::mlir::MemoryEffects::Write::get(), &operand,
                               ::mlir::SideEffects::DefaultResource::get());
      }
    }
  }];
}

class Poulpy_ReadWriteDpsOp<string mnemonic, list<Trait> traits = []> :
    Poulpy_DpsOp<mnemonic, traits, "Dst", 1, /*readsInit=*/1>;

def Poulpy_CtLike : TensorOrMemRef<[Poulpy_CiphertextType]>;
def Poulpy_PtLike : TensorOrMemRef<[Poulpy_PlaintextType]>;
def Poulpy_FloatLike : TensorOrMemRef<[AnyFloat]>;
def Poulpy_CtResult : Variadic<RankedTensorOf<[Poulpy_CiphertextType]>>;
def Poulpy_PtResult : Variadic<RankedTensorOf<[Poulpy_PlaintextType]>>;
def Poulpy_FloatResult : Variadic<RankedTensorOf<[AnyFloat]>>;

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

// maps to module.ckks_pt_vec_alloc + pt.set_meta + module.ckks_encode_reim_into
def Poulpy_EncodeOp : Poulpy_DpsOp<"encode", [], "Plaintext"> {
  let summary = "Encode real and imaginary slot vectors into a plaintext";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSEncodingOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_PtLike:$plaintext,
    Poulpy_FloatLike:$real,
    Poulpy_FloatLike:$imag,
    Poulpy_ScratchType:$scratch,
    I64Attr:$logDelta,
    I64Attr:$logBudget,
    I64Attr:$base2k
  );
  let results = (outs Poulpy_PtResult:$result);
}

// maps to module.ckks_decode_reim_into
def Poulpy_DecodeOp : Poulpy_DpsOp<"decode", [], "Real", 2> {
  let summary = "Decode a plaintext into real and imaginary slot vectors";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSEncodingOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_FloatLike:$real,
    Poulpy_FloatLike:$imag,
    Poulpy_PtLike:$plaintext,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_FloatResult:$result);
}

// maps to ckks_encrypt_sk
def Poulpy_EncryptOp : Poulpy_DpsOp<"encrypt", [], "Ciphertext"> {
  let summary = "Secret-key encryption of a CKKS plaintext";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSEncryptOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$ciphertext,
    Poulpy_PtLike:$plaintext,
    Poulpy_SecretKeyType:$secret_key,
    Poulpy_ScratchType:$scratch,
    I64Attr:$base2k,
    I64Attr:$ctk
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_decrypt
def Poulpy_DecryptOp : Poulpy_DpsOp<"decrypt", [], "Plaintext"> {
  let summary = "Secret-key decryption of a CKKS ciphertext";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSDecryptOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_PtLike:$plaintext,
    Poulpy_CtLike:$ciphertext,
    Poulpy_SecretKeyType:$secret_key,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_PtResult:$result);
}

// maps to ckks_add_into
def Poulpy_AddOp : Poulpy_DpsOp<"add"> {
  let summary = "Computes `dst = a + b`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_CtLike:$b,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_add_assign
def Poulpy_AddAssignOp : Poulpy_ReadWriteDpsOp<"add_assign"> {
  let summary = "Computes `dst += a`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_sub_into
def Poulpy_SubOp : Poulpy_DpsOp<"sub"> {
  let summary = "Computes `dst = a - b`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_CtLike:$b,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_sub_assign
def Poulpy_SubAssignOp : Poulpy_ReadWriteDpsOp<"sub_assign"> {
  let summary = "Computes `dst -= a`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_mul_into
def Poulpy_MulOp : Poulpy_DpsOp<"mul"> {
  let summary = "Computes `dst = a * b` using tensor-product keyswitching via `tsk`.";
  let description = [{
    A fused multiply, relinearize and rescale: the product is relinearized
    with `tsk` and comes out at the operands' `log_delta` (rescale is a no-op
    in Poulpy). A ciphertext-ciphertext CKKS multiplication, together with its
    relinearization and rescale, always lowers to this single op. It takes
    Poulpy's one-pass multiply-relinearize when the operands, `dst` and `tsk`
    share one `base2k` at rank 1.

    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSMulOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_CtLike:$b,
    Poulpy_TensorKeyType:$tsk,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_mul_assign
def Poulpy_MulAssignOp : Poulpy_ReadWriteDpsOp<"mul_assign"> {
  let summary = "Computes `dst *= a` in-place using tensor-product keyswitching via `tsk`.";
  let description = [{
    The in-place form of `poulpy.mul`: also a fused multiply, relinearize and
    rescale.

    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSMulOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_TensorKeyType:$tsk,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_add_pt_vec_into
def Poulpy_AddPtVecOp : Poulpy_DpsOp<"add_pt_vec"> {
  let summary = "Computes `dst = a + b`, where `b` is a plaintext.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_PtLike:$b,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_add_pt_vec_assign
def Poulpy_AddPtVecAssignOp : Poulpy_ReadWriteDpsOp<"add_pt_vec_assign"> {
  let summary = "Computes `dst += a`, where `a` is a plaintext.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSAddOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_PtLike:$a,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_sub_pt_vec_into
def Poulpy_SubPtVecOp : Poulpy_DpsOp<"sub_pt_vec"> {
  let summary = "Computes `dst = a - b`, where `b` is a plaintext.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_PtLike:$b,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_sub_pt_vec_assign
def Poulpy_SubPtVecAssignOp : Poulpy_ReadWriteDpsOp<"sub_pt_vec_assign"> {
  let summary = "Computes `dst -= a`, where `a` is a plaintext.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSSubOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_PtLike:$a,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_mul_pt_vec_into
def Poulpy_MulPtVecOp : Poulpy_DpsOp<"mul_pt_vec"> {
  let summary = "Computes `dst = a * b`, where `b` is a plaintext.";
  let description = [{
    Consumes the plaintext's `log_delta` bits of `log_budget`: Poulpy folds the
    plaintext in at its full declared width, so encode it at the precision to
    fold in.

    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSMulOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$a,
    Poulpy_PtLike:$b,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_mul_pt_vec_assign
def Poulpy_MulPtVecAssignOp : Poulpy_ReadWriteDpsOp<"mul_pt_vec_assign"> {
  let summary = "Computes `dst *= a`, where `a` is a plaintext.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSMulOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_PtLike:$a,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_neg_into
def Poulpy_NegOp : Poulpy_DpsOp<"neg"> {
  let summary = "Computes `dst = -src`.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSNegOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$src,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_neg_assign
def Poulpy_NegAssignOp : Poulpy_ReadWriteDpsOp<"neg_assign"> {
  let summary = "Computes `dst = -dst` in-place.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSNegOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_rotate_into
def Poulpy_RotateOp : Poulpy_DpsOp<"rotate"> {
  let summary = "Computes `dst = rotate(src, k)`: shifts all complex slots by `k` positions.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSRotateOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$src,
    I64Attr:$k,
    Poulpy_AutomorphismKeyMapType:$keys,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_rotate_assign
def Poulpy_RotateAssignOp : Poulpy_ReadWriteDpsOp<"rotate_assign"> {
  let summary = "Computes `dst = rotate(dst, k)` in-place. Metadata is unchanged.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSRotateOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    I64Attr:$k,
    Poulpy_AutomorphismKeyMapType:$keys,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_mul_pow2_into
def Poulpy_MulPow2Op : Poulpy_DpsOp<"mul_pow2"> {
  let summary = "Computes `dst = src * 2^bits`.";
  let description = [{
    Multiplies the encrypted message by `2^bits` at the same `log_delta`.

    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSPow2Ops.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$src,
    I64Attr:$bits,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_mul_pow2_assign
def Poulpy_MulPow2AssignOp : Poulpy_ReadWriteDpsOp<"mul_pow2_assign"> {
  let summary = "Computes `dst *= 2^bits` in-place.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSPow2Ops.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    I64Attr:$bits,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_div_pow2_into
def Poulpy_DivPow2Op : Poulpy_DpsOp<"div_pow2"> {
  let summary = "Computes `dst = src / 2^bits`.";
  let description = [{
    Divides the encrypted message by `2^bits`, moving `bits` from `log_budget`
    to `log_delta`. This is not a CKKS rescale, which keeps the message. In
    Poulpy a rescale is a no-op: a product already comes out at its operands'
    `log_delta`.

    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSPow2Ops.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$src,
    I64Attr:$bits,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_div_pow2_assign
def Poulpy_DivPow2AssignOp : Poulpy_ReadWriteDpsOp<"div_pow2_assign"> {
  let summary = "Computes `dst /= 2^bits` in-place.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSPow2Ops.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    I64Attr:$bits,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

// maps to ckks_copy
def Poulpy_CompactLimbsOp : Poulpy_DpsOp<"compact_limbs"> {
  let summary = "Level-aware ciphertext copy that compacts src into a smaller dst.";
  let description = [{
    See https://docs.rs/poulpy-ckks/latest/poulpy_ckks/api/trait.CKKSCopyOps.html
  }];
  let arguments = (ins
    Poulpy_ModuleType:$module,
    Poulpy_CtLike:$dst,
    Poulpy_CtLike:$src,
    Poulpy_ScratchType:$scratch
  );
  let results = (outs Poulpy_CtResult:$result);
}

#endif // LIB_DIALECT_POULPY_IR_POULPYOPS_TD_
