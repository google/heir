#include "lib/Target/Poulpy/PoulpyEmitter.h"

#include "lib/Dialect/Poulpy/IR/PoulpyDialect.h"
#include "lib/Dialect/Poulpy/IR/PoulpyOps.h"
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.h"
#include "lib/Target/Poulpy/PoulpyTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#define DEBUG_TYPE "poulpy-emitter"

namespace mlir {
namespace heir {
namespace poulpy {

namespace {
FailureOr<std::string> detectBackend(ModuleOp* op) {
  std::optional<PoulpyBackend> found;
  for (auto funcOp : op->getOps<func::FuncOp>()) {
    // TODO(mmoro): also check ModuleCreateOp results for backend information
    // (a "setup" function may create its own module rather than receive one
    // as an argument)
    for (Type argType : funcOp.getArgumentTypes()) {
      auto moduleType = dyn_cast<ModuleType>(argType);
      if (!moduleType) continue;
      PoulpyBackend backend = moduleType.getBackend();
      if (found.has_value() && found != backend) {
        return op->emitError("poulpy module contains multiple backends");
      }
      found = backend;
    }
  }
  switch (found.value_or(PoulpyBackend::FFT64Ref)) {
    case PoulpyBackend::FFT64Ref:
      return std::string("FFT64Ref");
    case PoulpyBackend::NTT4x30Ref:
      return std::string("NTT4x30Ref");
  }
  llvm_unreachable("unhandled PoulpyBackend");
}

std::string valueOrClonedValue(Value value,
                               SelectVariableNames* variableNames) {
  auto expression = variableNames->getNameForValue(value);
  if (isa<BlockArgument>(value)) {
    expression += ".clone()";
  }
  return expression;
}

std::string ref(Value value, SelectVariableNames* variableNames) {
  auto expression = variableNames->getNameForValue(value);
  return isa<BlockArgument>(value) ? "&*" + expression : "&" + expression;
}

std::string refMut(Value value, SelectVariableNames* variableNames) {
  auto expression = variableNames->getNameForValue(value);
  return isa<BlockArgument>(value) ? "&mut *" + expression
                                   : "&mut " + expression;
}
}  // namespace

void registerToPoulpyTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-poulpy", "translate the poulpy dialect to Rust code for poulpy",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToPoulpy(op, output);
      },
      [](DialectRegistry& registry) {
        registry.insert<func::FuncDialect, memref::MemRefDialect,
                        poulpy::PoulpyDialect>();
      });
}

LogicalResult translateToPoulpy(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  PoulpyEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult PoulpyEmitter::translateBlock(Block& block) {
  for (Operation& op : block.getOperations()) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult PoulpyEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp, AddOp,
                AddAssignOp, SubOp, SubAssignOp, MulOp, MulAssignOp, RotateOp,
                RotateAssignOp, RescaleOp, RescaleAssignOp, CompactLimbsOp,
                AddUnnormalizedOp, SubUnnormalizedOp, NormalizeOp, EncodeOp,
                DecodeOp, EncryptOp, DecryptOp, ModuleCreateOp, ScratchAllocOp,
                memref::AllocOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation& op) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }

  return success();
}

void PoulpyEmitter::computeMutatedValues(func::FuncOp funcOp) {
  mutatedValues.clear();

  funcOp.walk([&](Operation* op) {
    llvm::TypeSwitch<Operation&, void>(*op)
        .Case<AddOp, AddAssignOp, SubOp, SubAssignOp, MulOp, MulAssignOp,
              RotateOp, RotateAssignOp, RescaleOp, RescaleAssignOp,
              CompactLimbsOp, AddUnnormalizedOp, SubUnnormalizedOp>(
            [&](auto op) { mutatedValues.insert(op.getDst()); })
        .Case<DecodeOp>([&](DecodeOp op) {
          mutatedValues.insert(op.getReal());
          mutatedValues.insert(op.getImag());
        })
        .Case<EncryptOp>(
            [&](EncryptOp op) { mutatedValues.insert(op.getCiphertext()); })
        .Case<DecryptOp>(
            [&](DecryptOp op) { mutatedValues.insert(op.getPlaintext()); })
        .Default([&](Operation& op) {});
  });
}

void PoulpyEmitter::materializeIfPending(Value dst, Value module,
                                         Value layoutSource,
                                         bool useSemanticWidth) {
  if (pendingAllocs.erase(dst)) {
    // The unnormalized ops are the only ones whose dst is typed as an
    // unnormalized ciphertext, so the wrapper follows from the type.
    bool wrapUnnormalized = isa<UnnormalizedCiphertextType>(
        cast<MemRefType>(dst.getType()).getElementType());
    auto layoutName = variableNames->getNameForValue(layoutSource);
    auto dstName = variableNames->getNameForValue(dst);
    auto moduleName = variableNames->getNameForValue(module);
    os << "let mut " << dstName << " = ";
    if (wrapUnnormalized) os << "CtUnnorm::new(";
    os << moduleName << ".ckks_ciphertext_alloc(" << layoutName << ".base2k(), "
       << layoutName << (useSemanticWidth ? ".k())" : ".max_k())");
    if (wrapUnnormalized) os << ")";
    os << ";\n";
  }
}

// The _assign ops read-before-write dst, so it must already be materialized,
// normalize produces a brand new value, so its result must NOT be
// materialized yet
LogicalResult PoulpyEmitter::checkPendingState(Value dst, Operation* op,
                                               bool shouldBePending) {
  if (pendingAllocs.contains(dst) == shouldBePending) return success();

  if (shouldBePending) {
    return op->emitError(
        "normalize result must come from an unmaterialized memref.alloc");
  }
  InFlightDiagnostic diag =
      op->emitError("dst has not been initialized before this use");
  diag.attachNote(dst.getLoc()) << "allocated here but never written to first";
  return diag;
}

void PoulpyEmitter::emitEncoderIfNeeded(Value module) {
  if (encoderEmitted) return;
  encoderEmitted = true;
  os << "let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>("
     << variableNames->getNameForValue(module) << ".n() / 2)?;\n";
}

LogicalResult PoulpyEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  auto backend = detectBackend(&moduleOp);
  if (failed(backend)) {
    moduleOp.emitOpError("Error while detecting backend");
    return failure();
  }
  os << "type BE = " << backend.value() << ";\n";
  os << kTypeAliases << "\n";
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult PoulpyEmitter::printOperation(func::FuncOp funcOp) {
  computeMutatedValues(funcOp);
  encoderEmitted = false;
  sourceCounter = 0;
  os << "pub fn " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": ";
    bool isMutated = mutatedValues.contains(arg);
    if (failed(emitType(arg.getType(), /*isArg=*/true, isMutated))) {
      return funcOp.emitOpError()
             << "Failed to emit poulpy type " << arg.getType();
    }
    os << ",\n";
  }
  os.unindent();
  os << ") -> Result<";

  auto numResults = funcOp.getNumResults();
  if (numResults == 0) {
    os << "()";
  } else if (numResults == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result, /*isArg=*/false, /*isMutated=*/false))) {
      return funcOp.emitOpError() << "Failed to emit poulpy type " << result;
    }
  } else {
    auto types = commaSeparatedTypes(funcOp.getResultTypes(), [&](Type t) {
      return convertType(t, /*isArg=*/false, /*isMutated=*/false);
    });
    if (failed(types))
      return funcOp.emitOpError() << "Failed to emit poulpy result types";
    os << "(" << *types << ")";
  }

  os << "> {\n";
  os.indent();
  for (Block& block : funcOp.getBlocks()) {
    if (failed(translateBlock(block))) {
      return funcOp.emitOpError()
             << "Failed to translate block of func " << funcOp.getName();
    }
  }

  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult PoulpyEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() == 0) {
    os << "Ok(())\n";
  } else if (op.getNumOperands() == 1) {
    auto returnOperand = op.getOperands()[0];
    auto expression = valueOrClonedValue(returnOperand, variableNames);
    os << "Ok(" << expression << ")\n";
  } else {
    auto values = commaSeparatedValues(op.getOperands(), [&](Value v) {
      return valueOrClonedValue(v, variableNames);
    });
    os << "Ok((" << values << "))\n";
  }
  return success();
}

LogicalResult PoulpyEmitter::printOperation(func::CallOp op) {
  auto enclosingModuleOp = op->getParentOfType<ModuleOp>();
  auto calleeOp = enclosingModuleOp.lookupSymbol<func::FuncOp>(op.getCallee());

  computeMutatedValues(calleeOp);

  SmallVector<std::string> args;
  for (auto [operand, calleeArg] :
       llvm::zip(op.getOperands(), calleeOp.getArguments())) {
    if (isa<ScratchType>(calleeArg.getType()) ||
        mutatedValues.contains(calleeArg)) {
      args.push_back(refMut(operand, variableNames));
    } else {
      args.push_back(ref(operand, variableNames));
    }
  }

  auto argList = llvm::join(args, ", ");

  auto numResults = op.getNumResults();

  if (numResults == 0) {
    os << calleeOp.getName() << "(" << argList << ")?;\n";
  } else if (numResults == 1) {
    os << "let " << variableNames->getNameForValue(op.getResult(0)) << " = "
       << calleeOp.getName() << "(" << argList << ")?;\n";
  } else {
    auto names = commaSeparatedValues(op.getResults(), [&](Value v) {
      return variableNames->getNameForValue(v);
    });
    os << "let (" << names << ") = " << calleeOp.getName() << "(" << argList
       << ")?;\n";
  }

  return success();
}

void PoulpyEmitter::emitCall(Value module, StringRef rustFn, Value dst,
                             ArrayRef<std::string> args, Value scratch) {
  os << variableNames->getNameForValue(module) << "." << rustFn << "("
     << refMut(dst, variableNames);
  for (const std::string& arg : args) os << ", " << arg;
  if (scratch)
    os << ", &mut " << variableNames->getNameForValue(scratch) << ".borrow()";
  os << ")?;\n";
}

template <typename OpTy>
LogicalResult PoulpyEmitter::emitBinaryOp(OpTy op, StringRef rustFn,
                                          ArrayRef<std::string> extraArgs) {
  materializeIfPending(op.getDst(), op.getModule(), op.getA(),
                       /*useSemanticWidth=*/false);

  SmallVector<std::string> args = {ref(op.getA(), variableNames),
                                   ref(op.getB(), variableNames)};
  args.append(extraArgs.begin(), extraArgs.end());
  emitCall(op.getModule(), rustFn, op.getDst(), args, op.getScratch());
  return success();
}

template <typename OpTy>
LogicalResult PoulpyEmitter::emitBinaryAssignOp(
    OpTy op, StringRef rustFn, ArrayRef<std::string> extraArgs) {
  if (failed(checkPendingState(op.getDst(), op, /*shouldBePending=*/false)))
    return failure();

  SmallVector<std::string> args = {ref(op.getA(), variableNames)};
  args.append(extraArgs.begin(), extraArgs.end());
  emitCall(op.getModule(), rustFn, op.getDst(), args, op.getScratch());
  return success();
}

LogicalResult PoulpyEmitter::printOperation(AddOp addOp) {
  return emitBinaryOp(addOp, "ckks_add_into");
}

LogicalResult PoulpyEmitter::printOperation(AddAssignOp addAssignOp) {
  return emitBinaryAssignOp(addAssignOp, "ckks_add_assign");
}

LogicalResult PoulpyEmitter::printOperation(SubOp subOp) {
  return emitBinaryOp(subOp, "ckks_sub_into");
}

LogicalResult PoulpyEmitter::printOperation(SubAssignOp subAssignOp) {
  return emitBinaryAssignOp(subAssignOp, "ckks_sub_assign");
}

LogicalResult PoulpyEmitter::printOperation(MulOp mulOp) {
  return emitBinaryOp(mulOp, "ckks_mul_into",
                      {ref(mulOp.getTsk(), variableNames)});
}

LogicalResult PoulpyEmitter::printOperation(MulAssignOp mulAssignOp) {
  return emitBinaryAssignOp(mulAssignOp, "ckks_mul_assign",
                            {ref(mulAssignOp.getTsk(), variableNames)});
}

LogicalResult PoulpyEmitter::printOperation(AddUnnormalizedOp op) {
  return emitBinaryOp(op, "ckks_add_into_unnormalized");
}

LogicalResult PoulpyEmitter::printOperation(SubUnnormalizedOp op) {
  return emitBinaryOp(op, "ckks_sub_into_unnormalized");
}

LogicalResult PoulpyEmitter::printOperation(RotateOp rotateOp) {
  auto dst = rotateOp.getDst();
  auto src = rotateOp.getSrc();

  materializeIfPending(dst, rotateOp.getModule(), src,
                       /*useSemanticWidth=*/false);

  emitCall(rotateOp.getModule(), "ckks_rotate_into", dst,
           {ref(src, variableNames), std::to_string(rotateOp.getK()) + "i64",
            ref(rotateOp.getKeys(), variableNames)},
           rotateOp.getScratch());

  return success();
}

LogicalResult PoulpyEmitter::printOperation(RotateAssignOp rotateAssignOp) {
  auto dst = rotateAssignOp.getDst();

  if (failed(checkPendingState(dst, rotateAssignOp, /*shouldBePending=*/false)))
    return failure();

  emitCall(rotateAssignOp.getModule(), "ckks_rotate_assign", dst,
           {std::to_string(rotateAssignOp.getK()) + "i64",
            ref(rotateAssignOp.getKeys(), variableNames)},
           rotateAssignOp.getScratch());

  return success();
}

LogicalResult PoulpyEmitter::printOperation(RescaleOp rescaleOp) {
  auto dst = rescaleOp.getDst();
  auto src = rescaleOp.getSrc();

  materializeIfPending(dst, rescaleOp.getModule(), src,
                       /*useSemanticWidth=*/false);

  emitCall(
      rescaleOp.getModule(), "ckks_div_pow2_into", dst,
      {ref(src, variableNames), std::to_string(rescaleOp.getBits()) + "usize"},
      rescaleOp.getScratch());

  return success();
}

// NOTE: ckks_div_pow2_assign takes no scratch argument, unlike every other
// _assign op. The dialect op still carries a scratch operand for structural
// uniformity, it's just not part of the Rust call.
LogicalResult PoulpyEmitter::printOperation(RescaleAssignOp rescaleAssignOp) {
  auto dst = rescaleAssignOp.getDst();

  if (failed(
          checkPendingState(dst, rescaleAssignOp, /*shouldBePending=*/false)))
    return failure();

  emitCall(rescaleAssignOp.getModule(), "ckks_div_pow2_assign", dst,
           {std::to_string(rescaleAssignOp.getBits()) + "usize"});

  return success();
}

LogicalResult PoulpyEmitter::printOperation(CompactLimbsOp compactLimbsOp) {
  auto dst = compactLimbsOp.getDst();
  auto src = compactLimbsOp.getSrc();

  // Unlike every other _into op, dst is allocated with the semantic width
  // (.k()), not the allocated capacity (.max_k()).
  materializeIfPending(dst, compactLimbsOp.getModule(), src,
                       /*useSemanticWidth=*/true);

  emitCall(compactLimbsOp.getModule(), "ckks_copy", dst,
           {ref(src, variableNames)}, compactLimbsOp.getScratch());

  return success();
}

LogicalResult PoulpyEmitter::printOperation(NormalizeOp normalizeOp) {
  auto module = normalizeOp.getModule();
  auto res = normalizeOp.getRes();
  auto a = normalizeOp.getA();
  auto scratch = normalizeOp.getScratch();

  if (failed(checkPendingState(res, normalizeOp, /*shouldBePending=*/true)))
    return failure();
  pendingAllocs.erase(res);

  os << "let mut " << variableNames->getNameForValue(res) << " = "
     << valueOrClonedValue(a, variableNames) << ".normalize("
     << ref(module, variableNames) << ", &mut "
     << variableNames->getNameForValue(scratch) << ".borrow());\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(EncodeOp encodeOp) {
  auto module = encodeOp.getModule();
  auto plaintext = encodeOp.getPlaintext();
  auto real = encodeOp.getReal();
  auto imag = encodeOp.getImag();
  int64_t logDelta = encodeOp.getLogDelta();
  int64_t logBudget = encodeOp.getLogBudget();
  int64_t base2k = encodeOp.getBase2k();

  if (failed(checkPendingState(plaintext, encodeOp, /*shouldBePending=*/true)))
    return failure();
  pendingAllocs.erase(plaintext);

  emitEncoderIfNeeded(module);

  auto moduleName = variableNames->getNameForValue(module);
  auto ptName = variableNames->getNameForValue(plaintext);
  os << "let mut " << ptName << " = " << moduleName
     << ".ckks_pt_vec_alloc(Base2K(" << base2k << "u32), TorusPrecision("
     << (logDelta + logBudget) << "u32));\n";
  os << ptName << ".set_meta(CKKSMeta { log_delta: " << logDelta
     << "usize, log_sparsity: 0usize });\n";
  os << "encoder.encode_reim(" << refMut(plaintext, variableNames) << ", "
     << ref(real, variableNames) << ", " << ref(imag, variableNames) << ")?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(DecodeOp decodeOp) {
  auto module = decodeOp.getModule();
  auto real = decodeOp.getReal();
  auto imag = decodeOp.getImag();
  auto plaintext = decodeOp.getPlaintext();

  emitEncoderIfNeeded(module);

  os << "encoder.decode_reim(" << ref(plaintext, variableNames) << ", "
     << refMut(real, variableNames) << ", " << refMut(imag, variableNames)
     << ")?;\n";

  return success();
}

LogicalResult PoulpyEmitter::printOperation(EncryptOp encryptOp) {
  auto module = encryptOp.getModule();
  auto ciphertext = encryptOp.getCiphertext();
  auto secretKey = encryptOp.getSecretKey();
  uint64_t base2k = encryptOp.getBase2k();
  uint64_t ctK = encryptOp.getCtk();

  if (failed(
          checkPendingState(ciphertext, encryptOp, /*shouldBePending=*/true)))
    return failure();
  pendingAllocs.erase(ciphertext);

  auto moduleName = variableNames->getNameForValue(module);
  os << "let mut " << variableNames->getNameForValue(ciphertext) << " = "
     << moduleName << ".ckks_ciphertext_alloc(Base2K(" << base2k
     << "u32), TorusPrecision(" << ctK << "u32));\n";

  os << "let enc_layout" << sourceCounter
     << " = EncryptionLayout::new_from_default_sigma(GLWELayout {\n";
  os.indent();
  os << "n: " << moduleName << ".ring_degree(), base2k: Base2K(" << base2k
     << "u32), k: TorusPrecision(" << ctK
     << "u32), rank: " << variableNames->getNameForValue(secretKey)
     << ".rank(),\n";
  os.unindent();
  os << "})?;\n";

  // TODO(mmoro): These seeds are deterministic placeholders, not real
  // randomness. Thread a real CSPRNG seed once a client-interface pass
  // exists to supply one.
  os << "let mut source" << sourceCounter << " = Source::new([" << sourceCounter
     << "u8; 32]);\n";
  os << "let mut source" << (sourceCounter + 1) << " = Source::new(["
     << (sourceCounter + 1) << "u8; 32]);\n";

  emitCall(module, "ckks_encrypt_sk", ciphertext,
           {ref(encryptOp.getPlaintext(), variableNames),
            ref(secretKey, variableNames),
            "&enc_layout" + std::to_string(sourceCounter),
            "&mut source" + std::to_string(sourceCounter),
            "&mut source" + std::to_string(sourceCounter + 1)},
           encryptOp.getScratch());

  sourceCounter += 2;

  return success();
}

LogicalResult PoulpyEmitter::printOperation(DecryptOp decryptOp) {
  auto module = decryptOp.getModule();
  auto plaintext = decryptOp.getPlaintext();
  auto ciphertext = decryptOp.getCiphertext();

  if (failed(checkPendingState(plaintext, decryptOp, /*shouldBePending=*/true)))
    return failure();
  pendingAllocs.erase(plaintext);

  os << "let mut " << variableNames->getNameForValue(plaintext) << " = "
     << variableNames->getNameForValue(module)
     << ".ckks_pt_vec_alloc_from_infos(" << ref(ciphertext, variableNames)
     << ");\n";

  emitCall(module, "ckks_decrypt", plaintext,
           {ref(ciphertext, variableNames),
            ref(decryptOp.getSecretKey(), variableNames)},
           decryptOp.getScratch());

  return success();
}

LogicalResult PoulpyEmitter::printOperation(ModuleCreateOp moduleCreateOp) {
  os << "let " << variableNames->getNameForValue(moduleCreateOp.getModule())
     << " = Module::<BE>::new(" << moduleCreateOp.getN() << "u64);\n";
  return success();
}

LogicalResult PoulpyEmitter::printOperation(ScratchAllocOp scratchAllocOp) {
  os << "let mut "
     << variableNames->getNameForValue(scratchAllocOp.getScratch())
     << " = ScratchOwned::<BE>::alloc(" << scratchAllocOp.getSize()
     << "usize);\n";
  return success();
}

LogicalResult PoulpyEmitter::printOperation(memref::AllocOp allocOp) {
  MemRefType resultType = allocOp.getType();
  if (resultType.getElementType().isF64()) {
    // Unlike ciphertext buffers, an f64 vector's size is already known
    // statically from the type so we materialize immediately
    if (resultType.getRank() != 1 || resultType.isDynamicDim(0)) {
      return allocOp.emitOpError(
          "unsupported memref shape for f64 alloc (expected static rank-1)");
    }
    os << "let mut " << variableNames->getNameForValue(allocOp.getResult())
       << " = vec![0f64; " << resultType.getDimSize(0) << "];\n";
    return success();
  }
  pendingAllocs.insert(allocOp.getResult());
  return success();
}

FailureOr<std::string> PoulpyEmitter::convertType(Type type, bool isArg,
                                                  bool isMutated) {
  return llvm::TypeSwitch<Type&, FailureOr<std::string>>(type)
      .Case<ModuleType>([&](ModuleType) -> FailureOr<std::string> {
        return std::string(isArg ? "&Module<BE>" : "Module<BE>");
      })
      .Case<ScratchType>([&](ScratchType) -> FailureOr<std::string> {
        return std::string(isArg ? "&mut ScratchOwned<BE>"
                                 : "ScratchOwned<BE>");
      })
      .Case<MemRefType>([&](MemRefType memRefType) -> FailureOr<std::string> {
        Type elementType = memRefType.getElementType();
        if (elementType.isF64()) {
          if (memRefType.getRank() != 1 || memRefType.isDynamicDim(0)) {
            return failure();
          }
          return std::string(isMutated ? "&mut [f64]" : "&[f64]");
        }
        if (memRefType.getRank() != 0) return failure();
        if (isa<CiphertextType>(elementType)) {
          return std::string(isArg ? (isMutated ? "&mut Ct" : "&Ct") : "Ct");
        }
        if (isa<UnnormalizedCiphertextType>(elementType)) {
          return std::string(isArg ? (isMutated ? "&mut CtUnnorm" : "&CtUnnorm")
                                   : "CtUnnorm");
        }
        if (isa<PlaintextType>(elementType)) {
          return std::string(isArg ? (isMutated ? "&mut Pt" : "&Pt") : "Pt");
        }
        return failure();
      })
      .Case<SecretKeyType>([&](SecretKeyType) -> FailureOr<std::string> {
        return std::string("&Sk");
      })
      .Case<TensorKeyType>([&](TensorKeyType) -> FailureOr<std::string> {
        return std::string("&Tsk");
      })
      .Case<AutomorphismKeyMapType>(
          [&](AutomorphismKeyMapType) -> FailureOr<std::string> {
            return std::string("&Akm");
          })
      .Default([&](Type&) { return failure(); });
}

LogicalResult PoulpyEmitter::emitType(Type type, bool isArg, bool isMutated) {
  auto result = convertType(type, isArg, isMutated);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}
PoulpyEmitter::PoulpyEmitter(raw_ostream& os,
                             SelectVariableNames* variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace poulpy
}  // namespace heir
}  // namespace mlir
