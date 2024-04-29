#include "lib/Dialect/BGV/Transforms/AddClientInterface.h"

#include <cstddef>
#include <string>

#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

#define GEN_PASS_DEF_ADDCLIENTINTERFACE
#include "lib/Dialect/BGV/Transforms/Passes.h.inc"

FailureOr<lwe::RLWEParamsAttr> getRlweParmsFromFuncOp(func::FuncOp op) {
  lwe::RLWEParamsAttr rlweParams = nullptr;
  auto argTypes = op.getArgumentTypes();
  for (auto argTy : argTypes) {
    if (auto argCtTy = dyn_cast<lwe::RLWECiphertextType>(argTy)) {
      if (rlweParams && rlweParams != argCtTy.getRlweParams()) {
        return op.emitError() << "Func op has multiple distinct RLWE params"
                              << " but only 1 is currently supported per func.";
      }
      rlweParams = argCtTy.getRlweParams();
    }
  }
  if (!rlweParams) {
    return op.emitError() << "Func op has no RLWE ciphertext arguments.";
  }
  return rlweParams;
}

/// Generates an encryption func for one or more types.
LogicalResult generateEncryptionFunc(
    func::FuncOp op, const std::string &encFuncName, TypeRange encFuncArgTypes,
    TypeRange encFuncResultTypes, bool usePublicKey,
    lwe::RLWEParamsAttr rlweParams, ImplicitLocOpBuilder &builder) {
  // The enryption function converts each plaintext operand to its encrypted
  // form. We also have to add a public/secret key arg, and we put it at the
  // end to maintain zippability of the non-key args.
  Type encryptionKeyType =
      usePublicKey
          ? (Type)lwe::RLWEPublicKeyType::get(op.getContext(), rlweParams)
          : (Type)lwe::RLWESecretKeyType::get(op.getContext(), rlweParams);
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
    auto argTy = encFuncArgTypes[i];
    auto resultTy = encFuncResultTypes[i];

    // If the output is encrypted, we need to encode and encrypt
    if (auto resultCtTy = dyn_cast<lwe::RLWECiphertextType>(resultTy)) {
      auto plaintextTy = lwe::RLWEPlaintextType::get(
          op.getContext(), resultCtTy.getEncoding(),
          resultCtTy.getRlweParams().getRing(), argTy);
      auto encoded = builder.create<lwe::RLWEEncodeOp>(
          plaintextTy, encFuncOp.getArgument(i), resultCtTy.getEncoding(),
          resultCtTy.getRlweParams().getRing());
      auto encrypted = builder.create<bgv::EncryptOp>(
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
                                     lwe::RLWEParamsAttr rlweParams,
                                     ImplicitLocOpBuilder &builder) {
  Type decryptionKeyType =
      lwe::RLWESecretKeyType::get(op.getContext(), rlweParams);
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
    if (auto argCtTy = dyn_cast<lwe::RLWECiphertextType>(argTy)) {
      auto plaintextTy = lwe::RLWEPlaintextType::get(
          op.getContext(), argCtTy.getEncoding(),
          argCtTy.getRlweParams().getRing(), resultTy);
      auto decrypted = builder.create<bgv::DecryptOp>(
          plaintextTy, decFuncOp.getArgument(i), secretKey);
      auto decoded = builder.create<lwe::RLWEDecodeOp>(
          resultTy, decrypted.getResult(), argCtTy.getEncoding(),
          argCtTy.getRlweParams().getRing());
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
LogicalResult convertFunc(func::FuncOp op, bool usePublicKey,
                          bool oneValuePerHelperFn) {
  auto module = op->getParentOfType<ModuleOp>();
  auto rlweParamsResult = getRlweParmsFromFuncOp(op);
  if (failed(rlweParamsResult)) {
    return failure();
  }
  lwe::RLWEParamsAttr rlweParams = rlweParamsResult.value();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  if (!oneValuePerHelperFn) {
    std::string encFuncName("");
    llvm::raw_string_ostream encNameOs(encFuncName);
    encNameOs << op.getSymName() << "__encrypt";

    std::string decFuncName("");
    llvm::raw_string_ostream decNameOs(decFuncName);
    decNameOs << op.getSymName() << "__decrypt";

    SmallVector<Type> encFuncArgTypes;
    SmallVector<Type> encFuncResultTypes;
    auto argTypes = op.getArgumentTypes();

    for (auto argTy : argTypes) {
      if (auto argCtTy = dyn_cast<lwe::RLWECiphertextType>(argTy)) {
        encFuncArgTypes.push_back(argCtTy.getUnderlyingType());
        encFuncResultTypes.push_back(argCtTy);
        continue;
      }

      // For plaintext arguments, the function is a no-op
      encFuncArgTypes.push_back(argTy);
      encFuncResultTypes.push_back(argTy);
    }

    if (failed(generateEncryptionFunc(op, encFuncName, encFuncArgTypes,
                                      encFuncResultTypes, usePublicKey,
                                      rlweParams, builder))) {
      return failure();
    }

    // insertion point is inside encryption func, move back out
    builder.setInsertionPointToEnd(module.getBody());

    auto returnTypes = op.getResultTypes();
    SmallVector<Type> decFuncArgTypes;
    SmallVector<Type> decFuncResultTypes;
    for (auto returnTy : returnTypes) {
      if (auto returnCtTy = dyn_cast<lwe::RLWECiphertextType>(returnTy)) {
        decFuncArgTypes.push_back(returnCtTy);
        decFuncResultTypes.push_back(returnCtTy.getUnderlyingType());
        continue;
      }

      // For plaintext results, the function is a no-op
      decFuncArgTypes.push_back(returnTy);
      decFuncResultTypes.push_back(returnTy);
    }

    if (failed(generateDecryptionFunc(op, decFuncName, decFuncArgTypes,
                                      decFuncResultTypes, rlweParams,
                                      builder))) {
      return failure();
    }
    return success();
  }

  // Otherwise, we need one encryption function per argument and one decryption
  // function per return value. This is mainly to avoid complicated C++ codegen
  // when encrypting multiple inputs which requires out-params.
  for (auto val : op.getArguments()) {
    auto argTy = val.getType();
    if (auto argCtTy = dyn_cast<lwe::RLWECiphertextType>(argTy)) {
      std::string encFuncName("");
      llvm::raw_string_ostream encNameOs(encFuncName);
      encNameOs << op.getSymName() << "__encrypt__arg" << val.getArgNumber();
      if (failed(generateEncryptionFunc(
              op, encFuncName, {argCtTy.getUnderlyingType()}, {argCtTy},
              usePublicKey, rlweParams, builder))) {
        return failure();
      }
      // insertion point is inside func, move back out
      builder.setInsertionPointToEnd(module.getBody());
    }
  }

  ArrayRef<Type> returnTypes = op.getFunctionType().getResults();
  for (size_t i = 0; i < returnTypes.size(); ++i) {
    auto returnTy = returnTypes[i];
    if (auto returnCtTy = dyn_cast<lwe::RLWECiphertextType>(returnTy)) {
      std::string decFuncName("");
      llvm::raw_string_ostream encNameOs(decFuncName);
      encNameOs << op.getSymName() << "__decrypt__result" << i;
      if (failed(generateDecryptionFunc(op, decFuncName, {returnCtTy},
                                        {returnCtTy.getUnderlyingType()},
                                        rlweParams, builder))) {
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

  void runOnOperation() override {
    auto result =
        getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
          if (failed(convertFunc(op, usePublicKey, oneValuePerHelperFn))) {
            op->emitError("Failed to add client interface for func");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }
};
}  // namespace bgv
}  // namespace heir
}  // namespace mlir
