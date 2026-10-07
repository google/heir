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
    operands prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

// The CHEDDAR payload types are move-only and the C++ API is destination-passing
// (`ctx->Add(out, a, b)`). To match that, every payload-producing op is modeled
// as a destination-passing-style (DPS) op on builtin tensors: a scalar payload
// is a rank-0 `tensor<!cheddar.X>`, the trailing `$output` operand is the
// destination (init), and the result is tied 1:1 to it. The op is valid both
// in tensor form (one or more results, pre-bufferization) and in buffer form
// (operands are `memref<!cheddar.X>`, zero results, post one-shot-bufferize).
// `getDpsInitsMutable` points at the init operand named by `initName` (the
// `$output` of most ops; `decode` writes `$value`, `mad_unsafe` its
// `$accumulator` for a true in-place update).
class Cheddar_DpsOp<string mnemonic, list<Trait> traits = [],
                    string initName = "Output", int initCount = 1,
                    bit readsInit = 0> :
        Cheddar_Op<mnemonic, traits # [DestinationStyleOpInterface,
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
  // Operand-aware memory effects (linalg-style): only memref operands carry
  // effects, so the tensor form stays value-semantics/side-effect-free (and
  // DCE/CSE-able) while the bufferized memref form reports read/write. The DPS
  // init memref is written; other memref operands are read. This lets the stock
  // ownership-based buffer-deallocation analyze the bufferized cheddar ops and
  // free intermediate GPU buffers (at the end of their owning block; see
  // EraseDealloc in CheddarToEmitC). (Defined out-of-line: the declaration
  // comes from DeclareOpInterfaceMethods<MemoryEffectsOpInterface>.)
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

class Cheddar_ReadWriteDpsOp<string mnemonic, list<Trait> traits = [],
                             string initName = "Output",
                             int initCount = 1> :
    Cheddar_DpsOp<mnemonic, traits, initName, initCount,
                  /*readsInit=*/1>;

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
// After bufferization the result is a `memref<!cheddar.{context,user_interface}>`
// out-param. The compute, encrypt and decrypt functions take the borrowed
// handles, which lower to raw pointers (`Context<word>*`,
// `UserInterface<word>*`).
def Cheddar_ContextLike : TensorOrMemRef<[Cheddar_Context]>;
def Cheddar_UserInterfaceLike : TensorOrMemRef<[Cheddar_UserInterface]>;
def Cheddar_ContextResult : Variadic<RankedTensorOf<[Cheddar_Context]>>;
def Cheddar_UserInterfaceResult : Variadic<RankedTensorOf<[Cheddar_UserInterface]>>;

// A BootContext is an owning setup handle just like Context (CHEDDAR's
// `BootContext<word>::Create` returns a `std::shared_ptr<BootContext<word>>`;
// a borrowed `!cheddar.boot_context` lowers to `BootContext<word>*`);
// bootstrapping programs create one of these instead of a plain Context.
// `AnyOwningContextLike` accepts either, used by ops (e.g.
// create_user_interface) that work off whichever context the program selected.
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
    the destination buffer hint. Models `Context<word>::Create(params)`, which
    returns a `std::shared_ptr<Context<word>>`.
  }];
  let arguments = (ins Cheddar_Parameter:$params, Cheddar_ContextLike:$output);
  let results = (outs Cheddar_ContextResult:$result);
}

def Cheddar_CreateUserInterfaceOp : Cheddar_DpsOp<"create_user_interface"> {
  let summary = "Create a CHEDDAR UserInterface for key gen and encrypt/decrypt";
  let description = [{
    Creates a UserInterface<word> from an owning context (DPS). The result is the
    owning user_interface; `$output` is the destination buffer hint. Models
    `std::make_unique<UserInterface<word>>(ctx)`.
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
    buffer hint. Models
    `BootContext<word>::Create(params, BootParameter(params.max_level_, numCtsLevels, numStcLevels, logMessageRatio))`.

    `logMessageRatio` ~ log2(q0 / scale): the headroom between the bottom modulus
    and the message scale that CHEDDAR's EvalMod assumes. It governs bootstrap
    precision -- too small and large messages hit the sine-approximation domain
    edge, too large and the absolute noise floor (~ q0/scale) dominates small
    messages. When omitted, CHEDDAR's default applies.
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

def Cheddar_PrepareBootstrapOp : Cheddar_ReadWriteDpsOp<
    "prepare_bootstrap", [SameVariadicResultSize], "Ctx", 2> {
  let summary = "Run the one-time bootstrap precompute and rotation-key setup";
  let description = [{
    Performs the one-time bootstrapping preparation on a `!cheddar.boot_context`
    and threads the resulting rotation keys into the UserInterface. Both
    `$ctx` and `$ui` are read-write DPS destinations. Models the canonical
    CHEDDAR sequence:
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
  let results = (outs
    Cheddar_BootContextResult:$ctxResult,
    Cheddar_UserInterfaceResult:$uiResult
  );
}

def Cheddar_GetEncoderOp : Cheddar_Op<"get_encoder", [Pure]> {
  let summary = "Get the encoder from a CHEDDAR context";
  let description = [{
    Returns a borrowed reference to the context's encoder (`ctx->encoder_`).
    Lowers to `const Encoder<word>& enc = heir::getEncoder(ctx);`, a helper
    that `--cheddar-emitc-boundary` emits into the generated code.
  }];
  let arguments = (ins Cheddar_AnyContext:$ctx);
  let results = (outs Cheddar_Encoder:$encoder);
}

def Cheddar_GetEvkMapOp : Cheddar_Op<"get_evk_map", [Pure]> {
  let summary = "Get the evaluation key map from a UserInterface";
  let description = [{
    Returns the UserInterface's EvkMap. Lowers to
    `const EvkMap<word>& evk = ui->GetEvkMap();`.
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvkMap:$evkMap);
}

def Cheddar_GetMultKeyOp : Cheddar_Op<"get_mult_key", [Pure]> {
  let summary = "Get the multiplication evaluation key";
  let description = [{
    Returns the multiplication key from an evaluation-key map. Lowers to
    `const EvaluationKey<word>& key = evkMap.GetMultiplicationKey();`.
  }];
  let arguments = (ins Cheddar_EvkMap:$evkMap, Cheddar_AnyContext:$ctx);
  let results = (outs Cheddar_EvalKey:$key);
}

def Cheddar_PrepareRotKeyOp : Cheddar_ReadWriteDpsOp<
    "prepare_rot_key", [], "Ui"> {
  let summary = "Generate a rotation key for a given distance";
  let description = [{
    Models `ui->PrepareRotationKey(distance, maxLevel)`, which generates a
    rotation key, mutating the UserInterface in place. DPS: `$ui` is the in-place
    destination and the result is the updated UserInterface (threaded to the
    next setup op, or returned from a `__configure` built from these setup ops).
    Must be called before using rotation with that distance; `maxLevel` is the
    maximum ciphertext level at which the key will be used.
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
    primes are the CKKS `Q` limbs and the aux primes are the `P` limbs. Both the
    canonical level config, one main prime per level (`{{1, 0}, {2, 0}, ...}`),
    and the default encryption level, `#mainPrimes - 1`, are derived from
    `mainPrimes`. `logScale` is the log2 of the default CKKS scale. This is the
    single setup op that produces the `!cheddar.parameter` consumed by
    `cheddar.create_context`, so a `__configure` built from these setup ops is
    self-contained.

    Bootstrapping programs override the defaults: `defaultEncryptionLevel`
    pins the encryption level below the chain top (the boot-circuit primes sit
    above it), and `denseHammingWeight`/`sparseHammingWeight` set the secret-key
    hamming weights CHEDDAR's BootContext requires. Each is omitted for
    non-bootstrapping programs, in which case the encryption level defaults to
    `#mainPrimes - 1` and no `SetHammingWeight` call is made.
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
  let assemblyFormat = "prop-dict attr-dict `:` type($params)";
}

//===----------------------------------------------------------------------===//
// Encode / Encrypt / Decrypt operations
//===----------------------------------------------------------------------===//

def Cheddar_EncodeOp : Cheddar_DpsOp<"encode", [PlaintextEncodeOpInterface]> {
  let summary = "Encode a message vector into a CHEDDAR plaintext";
  let description = [{
    Encodes `message` into the plaintext `$output` at `level`. Lowers (modulo
    named temporaries) to `std::vector<Complex> msg(message, message + n);`
    followed by `encoder.Encode(pt, level, encoder.GetScale(level), msg);`,
    i.e. the message is copied into a `std::vector<Complex>` and encoded at
    CHEDDAR's canonical scale for `level`.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    Cheddar_FloatLike:$message,
    Cheddar_PtLike:$output,
    Builtin_IntegerAttr:$level,
    OptionalAttr<Builtin_IntegerAttr>:$logScale
  );
  let results = (outs Cheddar_PtResult:$result);
}

def Cheddar_EncodeConstantOp : Cheddar_DpsOp<"encode_constant"> {
  let summary = "Encode a scalar double into a CHEDDAR constant";
  let description = [{
    Encodes the scalar `value` into the constant `$output` at `level`. Lowers
    (modulo named temporaries) to
    `encoder.EncodeConstant(constant, level, encoder.GetScale(level), value)`.
    The result is in RNS form for efficient ciphertext-scalar ops; CHEDDAR
    derives the exact RNS scale from the target level.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    AnyFloat:$value,
    Cheddar_ConstLike:$output,
    Builtin_IntegerAttr:$level
  );
  let results = (outs Cheddar_ConstResult:$result);
}

def Cheddar_DecodeOp : Cheddar_DpsOp<"decode", [], "Value"> {
  let summary = "Decode a CHEDDAR plaintext back to a message vector";
  let description = [{
    Decodes `plaintext` into the destination buffer `value` (the DPS init),
    which the DPS interface verifier ties 1:1 to the `decoded` result (same
    type). Lowers to `encoder.Decode(msg, pt)` into a temporary
    `std::vector<Complex> msg`, then a loop storing `std::real(msg.at(i))` into
    the destination.
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
    Calls ui->Encrypt(ct, pt). Test-only operation. `$output` is the destination
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
    Calls ui->Decrypt(pt, ct). Test-only operation. `$output` is the destination
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
  let description = [{
    Calls context->Add(res, ct, pt) (CHEDDAR overloads `Add` on the type of its
    last argument).
  }];
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
  let description = [{
    Calls context->Sub(res, ct, pt).
  }];
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
  let description = [{
    Calls context->Mult(res, ct, pt).
  }];
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
  let description = [{
    Calls context->Add(res, ct, constant).
  }];
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
  let description = [{
    Calls context->Mult(res, ct, constant).
  }];
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
    Calls context->HRot(res, a, rot_key, distance) with
    `rot_key = evk.GetRotationKey(distance)`, looked up on the `$evk` EvkMap
    operand. Single fused GPU kernel. Supports both static and dynamic
    distances.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_EvkMap:$evk,
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
    Calls context->HRotAdd(res, a, b, rot_key, distance) with
    `rot_key = evk.GetRotationKey(distance)`, looked up on the `$evk` EvkMap
    operand.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_EvkMap:$evk,
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
    Calls context->HConj(res, a, conj_key) with
    `conj_key = evk.GetConjugationKey()`, looked up on the `$evk` EvkMap
    operand.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_EvkMap:$evk,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HConjAddOp : Cheddar_DpsOp<"hconj_add"> {
  let summary = "Fused conjugation + addition";
  let description = [{
    Computes res = conj(a) + b in a single fused GPU kernel.
    Calls context->HConjAdd(res, a, b, conj_key) with
    `conj_key = evk.GetConjugationKey()`, looked up on the `$evk` EvkMap
    operand.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_EvkMap:$evk,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$addend,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MadUnsafeOp : Cheddar_ReadWriteDpsOp<
    "mad_unsafe", [], "Accumulator"> {
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

    Requires a `!cheddar.boot_context` (not a plain `!cheddar.context`).
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
