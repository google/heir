#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_

include "CheddarDialect.td"
include "CheddarTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "lib/Dialect/HEIRInterfaces.td"

class Cheddar_Op<string mnemonic, list<Trait> traits = []> :
        Op<Cheddar_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

// The CHEDDAR payload types are move-only and the C++ API is destination-passing
// (`ctx->Add(out, a, b)`). To match that, every payload-producing op is modeled
// as a destination-passing-style (DPS) op on builtin tensors: a scalar payload
// is a rank-0 `tensor<!cheddar.X>`, the trailing `$output` operand is the
// destination (init), and the result is tied 1:1 to it. The op is valid both
// in tensor form (one result, pre-bufferization) and in buffer form (operands
// are `memref<!cheddar.X>`, zero results, post one-shot-bufferize). The shared
// `getDpsInitsMutable` points at the init operand named by `initName` (the
// `$output` of most ops; `decode` writes `$value`, `mad_unsafe` its
// `$accumulator` for a true in-place update).
class Cheddar_DpsOp<string mnemonic, list<Trait> traits = [],
                    string initName = "Output"> :
        Cheddar_Op<mnemonic, traits # [DestinationStyleOpInterface,
            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let extraClassDeclaration = [{
    // The single init operand as a (length-1) MutableOperandRange.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), get}] # initName # [{Mutable().getOperandNumber(), 1);
    }
  }];
  // Operand-aware memory effects (linalg-style): only memref operands carry
  // effects, so the tensor form stays value-semantics/side-effect-free (and
  // DCE/CSE-able) while the bufferized memref form reports read/write. The DPS
  // init memref is written; other memref operands are read. This lets the stock
  // ownership-based buffer-deallocation analyze the bufferized cheddar ops and
  // free intermediate GPU buffers at their last use. (Defined out-of-line: the
  // declaration comes from DeclareOpInterfaceMethods<MemoryEffectsOpInterface>.)
  let extraClassDefinition = [{
    void $cppClass::getEffects(::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
            &effects) {
      for (::mlir::OpOperand &operand : getOperation()->getOpOperands()) {
        if (!::llvm::isa<::mlir::MemRefType>(operand.get().getType()))
          continue;
        if (isDpsInit(&operand))
          effects.emplace_back(::mlir::MemoryEffects::Write::get(), &operand,
                               ::mlir::SideEffects::DefaultResource::get());
        else
          effects.emplace_back(::mlir::MemoryEffects::Read::get(), &operand,
                               ::mlir::SideEffects::DefaultResource::get());
      }
    }
  }];
}

// Payload operands and results are constrained by their cheddar element type.
// `*Like` accepts a tensor or memref (valid pre- and post-bufferization);
// `*Result` is a variadic ranked tensor (present in tensor form, absent in
// buffer form).
def Cheddar_CtLike : TensorOrMemRef<[Cheddar_Ciphertext]>;
def Cheddar_PtLike : TensorOrMemRef<[Cheddar_Plaintext]>;
def Cheddar_ConstLike : TensorOrMemRef<[Cheddar_Constant]>;
def Cheddar_FloatLike : TensorOrMemRef<[AnyFloat, AnyComplex]>;

def Cheddar_CtResult : Variadic<RankedTensorOf<[Cheddar_Ciphertext]>>;
def Cheddar_PtResult : Variadic<RankedTensorOf<[Cheddar_Plaintext]>>;
def Cheddar_ConstResult : Variadic<RankedTensorOf<[Cheddar_Constant]>>;

// Owning setup-handle destinations. The `create_*` setup ops are
// destination-passing like the payload ops: in tensor form they produce an SSA
// result (the owning context / user_interface, consumed downstream) and carry a
// trailing `$output` init operand that is the bufferization destination hint.
// After one-shot-bufferize + buffer-results-to-out-params the result becomes a
// `memref<!cheddar.{context,user_interface}>` out-param, which
// `cheddar-emitc-boundary` re-types to the owning C++ smart-pointer references
// (`std::shared_ptr<Context<word>>&` / `std::unique_ptr<UserInterface<word>>&`),
// distinguishing them from the borrowed raw-pointer handles the
// compute/encrypt/decrypt funcs take.
def Cheddar_ContextLike : TensorOrMemRef<[Cheddar_Context]>;
def Cheddar_UserInterfaceLike : TensorOrMemRef<[Cheddar_UserInterface]>;
def Cheddar_ContextResult : Variadic<RankedTensorOf<[Cheddar_Context]>>;
def Cheddar_UserInterfaceResult : Variadic<RankedTensorOf<[Cheddar_UserInterface]>>;

// A BootContext is an owning setup handle just like Context (it lowers to
// `std::shared_ptr<BootContext<word>>`); bootstrapping programs create one of
// these instead of a plain Context. `AnyOwningContextLike` accepts either, used
// by ops (e.g. create_user_interface) that work off whichever context the
// program selected.
def Cheddar_BootContextLike : TensorOrMemRef<[Cheddar_BootContext]>;
def Cheddar_BootContextResult : Variadic<RankedTensorOf<[Cheddar_BootContext]>>;
def Cheddar_AnyOwningContextLike :
    TensorOrMemRef<[Cheddar_Context, Cheddar_BootContext]>;

//===----------------------------------------------------------------------===//
// Setup operations
//===----------------------------------------------------------------------===//

def Cheddar_CreateContextOp : Cheddar_DpsOp<"create_context"> {
  let summary = "Create a CHEDDAR context from parameters";
  let description = [{
    Creates a CHEDDAR Context<word> from a Parameter object (DPS). The result is
    the owning context (the main server-side computation engine); `$output` is
    the destination buffer hint. After bufferization the destination is a
    `memref<!cheddar.context>` re-typed to `std::shared_ptr<Context<word>>&` at
    the emitc boundary, so a generated `__configure` hands the context back
    through an out-param: `ctx_out = Context<word>::Create(params)`.
  }];
  let arguments = (ins Cheddar_Parameter:$params, Cheddar_ContextLike:$output);
  let results = (outs Cheddar_ContextResult:$result);
}

def Cheddar_CreateUserInterfaceOp : Cheddar_DpsOp<"create_user_interface"> {
  let summary = "Create a CHEDDAR UserInterface for key gen and encrypt/decrypt";
  let description = [{
    Creates a UserInterface<word> from an owning context (DPS). The result is the
    owning user_interface; `$output` is the destination buffer hint (becomes a
    `std::unique_ptr<UserInterface<word>>&` out-param at the boundary):
    `ui_out = std::make_unique<UserInterface<word>>(ctx)`.
  }];
  let arguments = (ins Cheddar_AnyOwningContextLike:$ctx,
                       Cheddar_UserInterfaceLike:$output);
  let results = (outs Cheddar_UserInterfaceResult:$result);
}

def Cheddar_CreateBootContextOp : Cheddar_DpsOp<"create_boot_context"> {
  let summary = "Create a CHEDDAR bootstrapping context from parameters";
  let description = [{
    Creates a `BootContext<word>` from a Parameter object (DPS), for programs
    that bootstrap. Bootstrapping is parameterized by the CoeffToSlot (CtS) and
    SlotToCoeff (StC) level budgets -- a depth/rotation trade-off that mirrors
    OpenFHE's `levelBudgetEncode`/`levelBudgetDecode` -- carried here as the
    `numCtsLevels`/`numStcLevels` attributes. `$output` is the destination
    buffer hint; after bufferization the result becomes a
    `memref<!cheddar.boot_context>` re-typed to `std::shared_ptr<BootContext<word>>&`
    at the emitc boundary:
    `ctx_out = BootContext<word>::Create(params, BootParameter(params.max_level_, numCtsLevels, numStcLevels, logMessageRatio))`.

    `logMessageRatio` ~ log2(q0 / scale): the headroom between the bottom modulus
    and the message scale that CHEDDAR's EvalMod assumes. It governs bootstrap
    precision -- too small and large messages hit the sine-approximation domain
    edge, too large and the absolute noise floor (~ q0/scale) dominates small
    messages. When omitted, the emitter falls back to CHEDDAR's default.
  }];
  let arguments = (ins
    Cheddar_Parameter:$params,
    Builtin_IntegerAttr:$numCtsLevels,
    Builtin_IntegerAttr:$numStcLevels,
    OptionalAttr<Builtin_IntegerAttr>:$logMessageRatio,
    Cheddar_BootContextLike:$output
  );
  let results = (outs Cheddar_BootContextResult:$result);
}

def Cheddar_PrepareBootstrapOp : Cheddar_DpsOp<"prepare_bootstrap", [], "Ui"> {
  let summary = "Run the one-time bootstrap precompute and rotation-key setup";
  let description = [{
    Performs the one-time bootstrapping preparation on a `!cheddar.boot_context`
    and threads the resulting rotation keys into the UserInterface (DPS on
    `$ui`). Emits the canonical CHEDDAR sequence:
      ctx->PrepareEvalMod();
      ctx->PrepareEvalSpecialFFT(numSlots);
      EvkRequest req; ctx->AddRequiredRotations(req, numSlots);
      ui->PrepareRotationKey(req);
    `numSlots` is the number of slots the bootstrap refreshes (full-slot boot
    uses the context's slot count). Must run after the boot context + user
    interface are created and (for programs that also rotate outside boot) after
    the program's own `prepare_rot_key`s, so all keys land in one EvkMap.
  }];
  let arguments = (ins
    Cheddar_BootContextLike:$ctx,
    Cheddar_UserInterfaceLike:$ui,
    Builtin_IntegerAttr:$numSlots
  );
  let results = (outs Cheddar_UserInterfaceResult:$result);
}

def Cheddar_GetEncoderOp : Cheddar_Op<"get_encoder", [Pure]> {
  let summary = "Get the encoder from a CHEDDAR context";
  let description = [{
    Returns a reference to the context's encoder (context->encoder_).
  }];
  let arguments = (ins Cheddar_AnyContext:$ctx);
  let results = (outs Cheddar_Encoder:$encoder);
}

def Cheddar_GetEvkMapOp : Cheddar_Op<"get_evk_map", [Pure]> {
  let summary = "Get the evaluation key map from a UserInterface";
  let description = [{
    Returns the EvkMap from the UserInterface (ui.GetEvkMap()).
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvkMap:$evkMap);
}

def Cheddar_GetMultKeyOp : Cheddar_Op<"get_mult_key", [Pure]> {
  let summary = "Get the multiplication evaluation key";
  let description = [{
    Returns the multiplication key from the UserInterface
    (ui.GetMultiplicationKey()).
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvalKey:$key);
}

def Cheddar_PrepareRotKeyOp : Cheddar_DpsOp<"prepare_rot_key", [], "Ui"> {
  let summary = "Generate a rotation key for a given distance";
  let description = [{
    Calls `ui->PrepareRotationKey(distance, maxLevel)` to generate a rotation
    key, mutating the UserInterface in place. DPS: `$ui` is the in-place
    destination and the result is the updated UserInterface (threaded to the
    next setup op / the configure return). Must be called before using rotation
    with that distance; `maxLevel` is the maximum ciphertext level at which the
    key will be used.
  }];
  let arguments = (ins
    Cheddar_UserInterfaceLike:$ui,
    Builtin_IntegerAttr:$distance,
    Builtin_IntegerAttr:$maxLevel
  );
  let results = (outs Cheddar_UserInterfaceResult:$result);
}

def Cheddar_MakeParameterOp : Cheddar_Op<"make_parameter", [Pure]> {
  let summary = "Construct a CHEDDAR Parameter from CKKS scheme parameters";
  let description = [{
    Builds a `cheddar::Parameter<word>` from the CKKS modulus chain. The main
    primes are the CKKS `Q` limbs and the aux primes are the `P` limbs; the
    canonical level config is one-main-prime-per-level (`{{1, 0}, {2, 0}, ...}`)
    and the default encryption level is `#mainPrimes - 1`, both derived from
    `mainPrimes` by the emitter. `logScale` is the log2 of the default CKKS
    scale. This is the single setup op that produces the `!cheddar.parameter`
    consumed by `cheddar.create_context`, so a generated `__configure` is
    self-contained.

    Bootstrapping programs override the defaults: `defaultEncryptionLevel`
    pins the encryption level below the chain top (the boot-circuit primes sit
    above it), and `denseHammingWeight`/`sparseHammingWeight` set the secret-key
    hamming weights CHEDDAR's BootContext requires. Each is omitted (and the
    emitter falls back to `#mainPrimes - 1` / not emitting a SetHammingWeight
    call) for non-bootstrapping programs.
  }];
  let arguments = (ins
    Builtin_IntegerAttr:$logN,
    Builtin_IntegerAttr:$logScale,
    DenseI64ArrayAttr:$mainPrimes,
    DenseI64ArrayAttr:$auxPrimes,
    OptionalAttr<Builtin_IntegerAttr>:$defaultEncryptionLevel,
    OptionalAttr<Builtin_IntegerAttr>:$denseHammingWeight,
    OptionalAttr<Builtin_IntegerAttr>:$sparseHammingWeight
  );
  let results = (outs Cheddar_Parameter:$params);
  let assemblyFormat = "attr-dict `:` type($params)";
}

//===----------------------------------------------------------------------===//
// Encode / Encrypt / Decrypt operations
//===----------------------------------------------------------------------===//

def Cheddar_EncodeOp : Cheddar_DpsOp<"encode", [PlaintextEncodeOpInterface]> {
  let summary = "Encode a message vector into a CHEDDAR plaintext";
  let description = [{
    Calls encoder.Encode(pt, level, scale, message). The message is a vector
    of complex numbers (or reals). `scale` is the C++ `double` value used by
    CHEDDAR's Encoder. `$output` is the destination plaintext.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    Cheddar_FloatLike:$message,
    Cheddar_PtLike:$output,
    Builtin_IntegerAttr:$level,
    F64Attr:$scale
  );
  let results = (outs Cheddar_PtResult:$result);
}

def Cheddar_EncodeConstantOp : Cheddar_DpsOp<"encode_constant"> {
  let summary = "Encode a scalar double into a CHEDDAR constant";
  let description = [{
    Calls encoder.EncodeConstant(constant, level, scale, number).
    The result is in RNS form for efficient ciphertext-scalar ops.
    `scale` is the C++ `double` value used by CHEDDAR's Encoder. `$output` is
    the destination constant.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    AnyFloat:$value,
    Cheddar_ConstLike:$output,
    Builtin_IntegerAttr:$level,
    F64Attr:$scale
  );
  let results = (outs Cheddar_ConstResult:$result);
}

def Cheddar_DecodeOp : Cheddar_DpsOp<"decode", [], "Value"> {
  let summary = "Decode a CHEDDAR plaintext back to a message vector";
  let description = [{
    Calls encoder.Decode(message, pt). `value` is the destination buffer the
    decoded message is written into (the DPS init); the DPS interface verifier
    ties it 1:1 to the `decoded` result (same type).
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    Cheddar_PtLike:$plaintext,
    Cheddar_FloatLike:$value
  );
  let results = (outs Variadic<AnyRankedTensor>:$decoded);
}

def Cheddar_EncryptOp : Cheddar_DpsOp<"encrypt"> {
  let summary = "Encrypt a plaintext into a ciphertext";
  let description = [{
    Calls ui.Encrypt(ct, pt). Test-only operation. `$output` is the destination
    ciphertext.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_DecryptOp : Cheddar_DpsOp<"decrypt"> {
  let summary = "Decrypt a ciphertext into a plaintext";
  let description = [{
    Calls ui.Decrypt(pt, ct). Test-only operation. `$output` is the destination
    plaintext.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$output
  );
  let results = (outs Cheddar_PtResult:$result);
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

class Cheddar_BinaryCtCtOp<string mnemonic, list<Trait> traits = []>
    : Cheddar_DpsOp<mnemonic, traits> {
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$lhs,
    Cheddar_CtLike:$rhs,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_AddOp : Cheddar_BinaryCtCtOp<"add"> {
  let summary = "Add two ciphertexts";
  let description = [{
    Calls context->Add(res, a, b).
  }];
}

def Cheddar_SubOp : Cheddar_BinaryCtCtOp<"sub"> {
  let summary = "Subtract two ciphertexts";
  let description = [{
    Calls context->Sub(res, a, b).
  }];
}

def Cheddar_MultOp : Cheddar_BinaryCtCtOp<"mult", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts (tensor product, no relin/rescale)";
  let description = [{
    Calls context->Mult(res, a, b). Produces a degree-3 ciphertext.
    Does NOT include relinearization or rescaling.
  }];
}

def Cheddar_AddPlainOp : Cheddar_DpsOp<"add_plain"> {
  let summary = "Add a plaintext to a ciphertext";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_SubPlainOp : Cheddar_DpsOp<"sub_plain"> {
  let summary = "Subtract a plaintext from a ciphertext";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MultPlainOp : Cheddar_DpsOp<"mult_plain"> {
  let summary = "Multiply a ciphertext by a plaintext (no rescale)";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_AddConstOp : Cheddar_DpsOp<"add_const"> {
  let summary = "Add a constant to a ciphertext";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_ConstLike:$constant,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MultConstOp : Cheddar_DpsOp<"mult_const"> {
  let summary = "Multiply a ciphertext by a constant (no rescale)";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_ConstLike:$constant,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_NegOp : Cheddar_DpsOp<"neg"> {
  let summary = "Negate a ciphertext";
  let description = [{
    Calls context->Neg(res, a).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_RescaleOp : Cheddar_DpsOp<"rescale"> {
  let summary = "Rescale a ciphertext (drop one level)";
  let description = [{
    Calls context->Rescale(res, a). Reduces the ciphertext level by 1.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_LevelDownOp : Cheddar_DpsOp<"level_down"> {
  let summary = "Reduce ciphertext to a target level";
  let description = [{
    Calls context->LevelDown(res, a, target_level).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output,
    Builtin_IntegerAttr:$targetLevel
  );
  let results = (outs Cheddar_CtResult:$result);
}

//===----------------------------------------------------------------------===//
// Key-switching operations
//===----------------------------------------------------------------------===//

def Cheddar_RelinearizeOp : Cheddar_DpsOp<"relinearize"> {
  let summary = "Relinearize a ciphertext (without rescale)";
  let description = [{
    Calls context->Relinearize(res, a, mult_key).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvalKey:$multKey,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_RelinearizeRescaleOp : Cheddar_DpsOp<"relinearize_rescale"> {
  let summary = "Fused relinearize + rescale";
  let description = [{
    Calls context->RelinearizeRescale(res, a, mult_key). Faster than
    separate Relinearize + Rescale.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvalKey:$multKey,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

//===----------------------------------------------------------------------===//
// Compound (fused) operations -- high-performance GPU kernels
//===----------------------------------------------------------------------===//

def Cheddar_HMultOp : Cheddar_DpsOp<"hmult", [IncreasesMulDepthOpInterface]> {
  let summary = "Fused multiply + relinearize (+ optional rescale)";
  let description = [{
    Calls context->HMult(res, a, b, mult_key, rescale).
    Single fused GPU kernel launch. The `rescale` attribute controls whether
    rescaling is included (default: true).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$lhs,
    Cheddar_CtLike:$rhs,
    Cheddar_EvalKey:$multKey,
    Cheddar_CtLike:$output,
    DefaultValuedAttr<BoolAttr, "true">:$rescale
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HRotOp : Cheddar_DpsOp<"hrot", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Fused key-switch + rotation";
  let description = [{
    Calls context->HRot(res, a, rot_key, distance). The rotation key is looked
    up from the `$ui` UserInterface operand using `distance`.
    Single fused GPU kernel. Supports both static and dynamic distances.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_UserInterface:$ui,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output,
    Optional<AnySignlessIntegerOrIndex>:$dynamic_distance,
    OptionalAttr<Builtin_IntegerAttr>:$static_distance
  );
  let results = (outs Cheddar_CtResult:$result);
  let hasVerifier = 1;
}

def Cheddar_HRotAddOp : Cheddar_DpsOp<"hrot_add", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Fused rotation + addition";
  let description = [{
    Computes res = rotate(a, distance) + b in a single fused GPU kernel.
    Calls context->HRotAdd(res, a, b, rot_key, distance). The rotation key
    is looked up from the `$ui` UserInterface operand using `distance`.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_UserInterface:$ui,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$addend,
    Cheddar_CtLike:$output,
    Builtin_IntegerAttr:$distance
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HConjOp : Cheddar_DpsOp<"hconj"> {
  let summary = "Fused key-switch + conjugation";
  let description = [{
    Calls context->HConj(res, a, conj_key). The conjugation key is looked
    up from the `$ui` UserInterface operand.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_UserInterface:$ui,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HConjAddOp : Cheddar_DpsOp<"hconj_add"> {
  let summary = "Fused conjugation + addition";
  let description = [{
    Computes res = conj(a) + b in a single fused GPU kernel.
    Calls context->HConjAdd(res, a, b, conj_key). The conjugation key is
    looked up from the `$ui` UserInterface operand.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_UserInterface:$ui,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$addend,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MadUnsafeOp : Cheddar_DpsOp<"mad_unsafe", [], "Accumulator"> {
  let summary = "Fused multiply-accumulate with constant (no rescale)";
  let description = [{
    Computes res += a * constant (in-place accumulation).
    Calls context->MadUnsafe(res, a, constant). The `$accumulator` operand is
    the in-place destination (the DPS init); the result is tied to it.

    Note: this op would typically be called `mac`,
    the current name reflects the spelling in Cheddar
    (`MadUnsafe`).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$accumulator,
    Cheddar_CtLike:$input,
    Cheddar_ConstLike:$constant
  );
  let results = (outs Cheddar_CtResult:$result);
}

//===----------------------------------------------------------------------===//
// Extension operations (bootstrapping, linear transforms, poly eval)
//===----------------------------------------------------------------------===//

def Cheddar_BootOp : Cheddar_DpsOp<"boot", [ResetsMulDepthOpInterface]> {
  let summary = "Bootstrap a ciphertext";
  let description = [{
    Calls boot_ctx->Boot(res, input, evk_map).
    Refreshes the ciphertext noise budget.

    Requires a `!cheddar.boot_context` (not a plain `!cheddar.context`): `Boot`
    is a BootContext method, so this op statically guarantees the context it
    runs on is a BootContext and the emitter needs no downcast.
  }];
  let arguments = (ins
    Cheddar_BootContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvkMap:$evkMap,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_LinearTransformOp : Cheddar_DpsOp<"linear_transform", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Apply a linear transform on a ciphertext";
  let description = [{
    Applies a matrix-vector product using CHEDDAR's LinearTransform extension
    with BSGS optimization and hoisting.

    The `diagonals` input is a 2D tensor where each row is a non-zero diagonal.
    The `diagonal_indices` attribute specifies which diagonal each row represents.
    The `level` attribute specifies the modulus level for the operation.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvkMap:$evkMap,
    TensorOrMemRef<[AnyFloat]>:$diagonals,
    Cheddar_CtLike:$output,
    DenseI32ArrayAttr:$diagonal_indices,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$bs,
    Builtin_IntegerAttr:$gs
  );
  let results = (outs Cheddar_CtResult:$result);
  let hasVerifier = 1;
}

def Cheddar_EvalPolyOp : Cheddar_DpsOp<"eval_poly"> {
  let summary = "Evaluate a polynomial on a ciphertext";
  let description = [{
    Evaluates a polynomial (e.g., Chebyshev approximation) on an encrypted
    input using CHEDDAR's EvalPoly extension.
    `level` is the level the input ciphertext enters at; `outputLevel` is the
    level the result lands at after the polynomial's multiplicative depth.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvkMap:$evkMap,
    Cheddar_CtLike:$output,
    F64ArrayAttr:$coefficients,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$outputLevel
  );
  let results = (outs Cheddar_CtResult:$result);
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_
