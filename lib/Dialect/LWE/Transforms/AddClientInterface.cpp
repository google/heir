#include "lib/Dialect/LWE/Transforms/AddClientInterface.h"

#include <cstddef>
#include <string>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDCLIENTINTERFACE
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

FailureOr<mlir::heir::polynomial::RingAttr> getEncRingFromFuncOp(
    func::FuncOp op) {
  mlir::heir::polynomial::RingAttr ring = nullptr;
  auto argTypes = op.getArgumentTypes();
  for (auto argTy : argTypes) {
    // Strip containers (tensor/etc)
    argTy = getElementTypeOrSelf(argTy);
    // Check that parameters are unique
    if (auto argCtTy = dyn_cast<lwe::NewLWECiphertextType>(argTy)) {
      if (ring && ring != argCtTy.getCiphertextSpace().getRing()) {
        return op.emitError() << "Func op has multiple distinct RLWE params"
                              << " but only 1 is currently supported per func.";
      }
      ring = argCtTy.getCiphertextSpace().getRing();
    }
  }
  if (!ring) {
    return op.emitError() << "Func op has no RLWE ciphertext arguments.";
  }
  return ring;
}

FailureOr<mlir::heir::polynomial::RingAttr> getDecRingFromFuncOp(
    func::FuncOp op) {
  mlir::heir::polynomial::RingAttr ring = nullptr;
  auto resultTypes = op.getFunctionType().getResults();
  for (auto resultTy : resultTypes) {
    // Strip containers (tensor/etc)
    resultTy = getElementTypeOrSelf(resultTy);
    // Check that parameters are unique
    if (auto resultCtTy = dyn_cast<lwe::NewLWECiphertextType>(resultTy)) {
      if (ring && ring != resultCtTy.getCiphertextSpace().getRing()) {
        return op.emitError() << "Func op has multiple distinct RLWE params"
                              << " but only 1 is currently supported per func.";
      }
      ring = resultCtTy.getCiphertextSpace().getRing();
    }
  }
  if (!ring) {
    return op.emitError() << "Func op has no RLWE ciphertext results.";
  }
  return ring;
}

/// Generates an encryption func for one or more types.
LogicalResult generateEncryptionFunc(
    func::FuncOp op, const std::string &encFuncName, TypeRange encFuncArgTypes,
    TypeRange encFuncResultTypes, bool usePublicKey,
    mlir::heir::polynomial::RingAttr ring, ImplicitLocOpBuilder &builder) {
  // The enryption function converts each plaintext operand to its encrypted
  // form. We also have to add a public/secret key arg, and we put it at the
  // end to maintain zippability of the non-key args.
  auto *ctx = op->getContext();
  Type encryptionKeyType =
      usePublicKey ? (Type)lwe::NewLWEPublicKeyType::get(
                         op.getContext(), KeyAttr::get(ctx, 0), ring)
                   : (Type)lwe::NewLWESecretKeyType::get(
                         op.getContext(), KeyAttr::get(ctx, 0), ring);
  SmallVector<Type> funcArgTypes(encFuncArgTypes.begin(),
                                 encFuncArgTypes.end());
  funcArgTypes.push_back(encryptionKeyType);

  FunctionType encFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, encFuncResultTypes);
  auto encFuncOp = builder.create<func::FuncOp>(encFuncName, encFuncType);
  Block *entryBlock = encFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  Value secretKey = encFuncOp.getArgument(encFuncOp.getNumArguments() - 1);

  SmallVector<Value> encValuesToReturn;
  // TODO(#615): encode/decode should convert scalar types to tensors.
  for (size_t i = 0; i < encFuncArgTypes.size(); ++i) {
    auto resultTy = encFuncResultTypes[i];

    // If the output is encrypted, we need to encode and encrypt
    if (auto resultCtTy = dyn_cast<lwe::NewLWECiphertextType>(resultTy)) {
      auto plaintextTy = lwe::NewLWEPlaintextType::get(
          op.getContext(), resultCtTy.getApplicationData(),
          resultCtTy.getPlaintextSpace());
      auto encoded = builder.create<lwe::RLWEEncodeOp>(
          plaintextTy, encFuncOp.getArgument(i),
          resultCtTy.getPlaintextSpace().getEncoding(),
          resultCtTy.getPlaintextSpace().getRing());
      auto encrypted = builder.create<lwe::RLWEEncryptOp>(
          resultCtTy, encoded.getResult(), secretKey);
      encValuesToReturn.push_back(encrypted.getResult());
      continue;
    }

    // Otherwise, return the input unchanged.
    encValuesToReturn.push_back(encFuncOp.getArgument(i));
  }

  builder.create<func::ReturnOp>(encValuesToReturn);
  return success();
}

/// Generates a decryption func for one or more types.
LogicalResult generateDecryptionFunc(func::FuncOp op,
                                     const std::string &decFuncName,
                                     TypeRange decFuncArgTypes,
                                     TypeRange decFuncResultTypes,
                                     mlir::heir::polynomial::RingAttr ring,
                                     ImplicitLocOpBuilder &builder) {
  Type decryptionKeyType = lwe::NewLWESecretKeyType::get(
      op.getContext(), KeyAttr::get(op.getContext(), 0), ring);
  SmallVector<Type> funcArgTypes(decFuncArgTypes.begin(),
                                 decFuncArgTypes.end());
  funcArgTypes.push_back(decryptionKeyType);

  // Then the decryption function
  FunctionType decFuncType =
      FunctionType::get(builder.getContext(), funcArgTypes, decFuncResultTypes);
  auto decFuncOp = builder.create<func::FuncOp>(decFuncName, decFuncType);
  builder.setInsertionPointToEnd(decFuncOp.addEntryBlock());
  Value secretKey = decFuncOp.getArgument(decFuncOp.getNumArguments() - 1);

  SmallVector<Value> decValuesToReturn;
  // Use result types because arg types has the secret key at the end, but
  // result types does not
  for (size_t i = 0; i < decFuncResultTypes.size(); ++i) {
    auto argTy = decFuncArgTypes[i];
    auto resultTy = decFuncResultTypes[i];

    // If the input is ciphertext, we need to decode and decrypt
    if (auto argCtTy = dyn_cast<lwe::NewLWECiphertextType>(argTy)) {
      auto plaintextTy = lwe::NewLWEPlaintextType::get(
          op.getContext(), argCtTy.getApplicationData(),
          argCtTy.getPlaintextSpace());
      auto decrypted = builder.create<lwe::RLWEDecryptOp>(
          plaintextTy, decFuncOp.getArgument(i), secretKey);
      auto decoded = builder.create<lwe::RLWEDecodeOp>(
          resultTy, decrypted.getResult(),
          argCtTy.getPlaintextSpace().getEncoding(),
          argCtTy.getPlaintextSpace().getRing());
      // FIXME: if the input is a scalar type, we must add a tensor.extract op.
      // The decode op's tablegen should also support having tensor types as
      // outputs if it doesn't already.
      decValuesToReturn.push_back(decoded.getResult());
      continue;
    }

    // Otherwise, return the input unchanged.
    decValuesToReturn.push_back(decFuncOp.getArgument(i));
  }

  builder.create<func::ReturnOp>(decValuesToReturn);
  return success();
}

/// Adds the client interface for a single func. This should only be used on the
/// "entry" func for the IR being compiled, but there may be multiple.
LogicalResult convertFunc(func::FuncOp op, bool usePublicKey) {
  auto module = op->getParentOfType<ModuleOp>();
  auto ringEncResult = getEncRingFromFuncOp(op);
  auto ringDecResult = getDecRingFromFuncOp(op);
  if (failed(ringEncResult) || failed(ringDecResult)) {
    return failure();
  }
  mlir::heir::polynomial::RingAttr ringEnc = ringEncResult.value();
  mlir::heir::polynomial::RingAttr ringDec = ringDecResult.value();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  // We need one encryption function per argument and one decryption
  // function per return value. This is mainly to avoid complicated C++ codegen
  // when encrypting multiple inputs which requires out-params.
  for (auto val : op.getArguments()) {
    auto argTy = val.getType();
    if (auto argCtTy = dyn_cast<lwe::NewLWECiphertextType>(argTy)) {
      std::string encFuncName("");
      llvm::raw_string_ostream encNameOs(encFuncName);
      encNameOs << op.getSymName() << "__encrypt__arg" << val.getArgNumber();
      if (failed(generateEncryptionFunc(
              op, encFuncName, {argCtTy.getApplicationData().getMessageType()},
              {argCtTy}, usePublicKey, ringEnc, builder))) {
        return failure();
      }
      // insertion point is inside func, move back out
      builder.setInsertionPointToEnd(module.getBody());
    }
  }

  ArrayRef<Type> returnTypes = op.getFunctionType().getResults();
  for (size_t i = 0; i < returnTypes.size(); ++i) {
    auto returnTy = returnTypes[i];
    if (auto returnCtTy = dyn_cast<lwe::NewLWECiphertextType>(returnTy)) {
      std::string decFuncName("");
      llvm::raw_string_ostream encNameOs(decFuncName);
      encNameOs << op.getSymName() << "__decrypt__result" << i;
      if (failed(generateDecryptionFunc(
              op, decFuncName, {returnCtTy},
              {returnCtTy.getApplicationData().getMessageType()}, ringDec,
              builder))) {
        return failure();
      }
      // insertion point is inside func, move back out
      builder.setInsertionPointToEnd(module.getBody());
    }
  }

  return success();
}

struct AddClientInterface : impl::AddClientInterfaceBase<AddClientInterface> {
  using AddClientInterfaceBase::AddClientInterfaceBase;

  bool getUsePublicKey() {
    bool usePublicKey = true;
    if (moduleIsBGVOrBFV(getOperation())) {
      auto bgvSchemeParamAttr =
          getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
              bgv::BGVDialect::kSchemeParamAttrName);
      if (!bgvSchemeParamAttr) {
        getOperation()->emitError("No BGV scheme param found.");
        signalPassFailure();
      }
      usePublicKey =
          bgvSchemeParamAttr.getEncryptionType() == bgv::BGVEncryptionType::pk;
    } else if (moduleIsCKKS(getOperation())) {
      auto ckksSchemeParamAttr =
          getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
              ckks::CKKSDialect::kSchemeParamAttrName);
      if (!ckksSchemeParamAttr) {
        getOperation()->emitError("No CKKS scheme param found.");
        signalPassFailure();
      }
      usePublicKey = ckksSchemeParamAttr.getEncryptionType() ==
                     ckks::CKKSEncryptionType::pk;
    } else {
      getOperation()->emitError("Unsupported RLWE scheme");
      signalPassFailure();
    }
    return usePublicKey;
  }

  void runOnOperation() override {
    auto result =
        getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
          if (op.isDeclaration()) {
            op->emitWarning("Skipping client interface for external func");
            return WalkResult::advance();
          }
          if (failed(convertFunc(op, getUsePublicKey()))) {
            op->emitError("Failed to add client interface for func");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
