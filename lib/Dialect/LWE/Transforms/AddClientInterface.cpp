#include "lib/Dialect/LWE/Transforms/AddClientInterface.h"

#include <cstddef>
#include <string>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "lwe-add-client-interface"

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDCLIENTINTERFACE
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

using ::mlir::heir::polynomial::RingAttr;
using ::mlir::heir::tensor_ext::OriginalTypeAttr;

auto &kOriginalTypeAttrName =
    tensor_ext::TensorExtDialect::kOriginalTypeAttrName;

int64_t getNumSlots(Operation *op) {
  return llvm::TypeSwitch<Attribute, int64_t>(getSchemeParamAttr(op))
      .Case<bgv::SchemeParamAttr>(
          // The numSlots is multiplied by 2 to get the param degree
          // so we have to halve it.
          [](auto attr) -> int64_t { return (1L << attr.getLogN()) / 2; })
      .Case<ckks::SchemeParamAttr>(
          [](auto attr) -> int64_t { return (1L << attr.getLogN()) / 2; })
      .Default([](Attribute attr) -> int64_t {
        assert(false && "Unsupported scheme parame attribute");
        return 0;
      });
}

FailureOr<RingAttr> getEncRingFromFuncOp(func::FuncOp op) {
  RingAttr ring = nullptr;
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

FailureOr<RingAttr> getDecRingFromFuncOp(func::FuncOp op) {
  RingAttr ring = nullptr;
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
    TypeRange encFuncResultTypes, bool usePublicKey, RingAttr ring,
    ArrayRef<OriginalTypeAttr> originalTypes, ImplicitLocOpBuilder &builder) {
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
  for (size_t i = 0; i < encFuncArgTypes.size(); ++i) {
    auto resultTy = encFuncResultTypes[i];
    auto originalType = originalTypes[i];
    auto operand = encFuncOp.getArgument(i);

    // If the output is encrypted, we need to encode and encrypt
    if (auto resultCtTy = dyn_cast<lwe::NewLWECiphertextType>(resultTy)) {
      // Apply the layout from the original type, which handles the job of
      // converting a scalar to a tensor, and out a tensor in a
      // ciphertext-semantic tensor so that it can then be encoded/encrypted.
      // Note that this op still needs to be lowered, which implies the
      // relevant pattern from convert-to-ciphertext-semantics must be run
      // after this pass.
      auto assignLayoutOp = builder.create<tensor_ext::AssignLayoutOp>(
          op.getLoc(), operand, originalType.getLayout());

      auto plaintextTy = lwe::NewLWEPlaintextType::get(
          op.getContext(), resultCtTy.getApplicationData(),
          resultCtTy.getPlaintextSpace());
      auto encoded = builder.create<lwe::RLWEEncodeOp>(
          plaintextTy, assignLayoutOp.getResult(),
          resultCtTy.getPlaintextSpace().getEncoding(),
          resultCtTy.getPlaintextSpace().getRing());
      auto encrypted = builder.create<lwe::RLWEEncryptOp>(
          resultCtTy, encoded.getResult(), secretKey);
      encValuesToReturn.push_back(encrypted.getResult());
      continue;
    }

    // Otherwise, return the input unchanged.
    encValuesToReturn.push_back(operand);
  }

  builder.create<func::ReturnOp>(encValuesToReturn);
  return success();
}

/// Generates a decryption func for one or more types.
LogicalResult generateDecryptionFunc(
    func::FuncOp op, const std::string &decFuncName, TypeRange decFuncArgTypes,
    TypeRange decFuncResultTypes, RingAttr ring,
    ArrayRef<OriginalTypeAttr> originalTypes, ImplicitLocOpBuilder &builder) {
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

  int64_t numSlots = getNumSlots(op);
  SmallVector<Value> decValuesToReturn;
  // Use result types because arg types has the secret key at the end, but
  // result types does not
  for (size_t i = 0; i < decFuncResultTypes.size(); ++i) {
    auto argTy = decFuncArgTypes[i];
    auto originalTypeAttr = originalTypes[i];

    // If the input is ciphertext, we need to decode and decrypt
    if (auto argCtTy = dyn_cast<lwe::NewLWECiphertextType>(argTy)) {
      auto plaintextTy = lwe::NewLWEPlaintextType::get(
          op.getContext(), argCtTy.getApplicationData(),
          argCtTy.getPlaintextSpace());
      auto decrypted = builder.create<lwe::RLWEDecryptOp>(
          plaintextTy, decFuncOp.getArgument(i), secretKey);

      Type dataSemanticType = originalTypeAttr.getOriginalType();
      RankedTensorType ciphertextSemanticType = cast<RankedTensorType>(
          isa<RankedTensorType>(dataSemanticType)
              ? materializeLayout(cast<RankedTensorType>(dataSemanticType),
                                  originalTypeAttr.getLayout(), numSlots)
              : materializeScalarLayout(
                    dataSemanticType, originalTypeAttr.getLayout(), numSlots));
      auto decoded = builder.create<lwe::RLWEDecodeOp>(
          ciphertextSemanticType, decrypted.getResult(),
          argCtTy.getPlaintextSpace().getEncoding(),
          argCtTy.getPlaintextSpace().getRing());
      Value unpacked;

      // TODO(#1677): support more complex unpacking
      if (isa<RankedTensorType>(dataSemanticType)) {
        auto tensorTy = cast<RankedTensorType>(dataSemanticType);
        if (tensorTy.getRank() > 1) {
          return op.emitError() << "Decryption function does not yet support "
                                   "tensors of rank > 1";
        }
        if (tensorTy == decoded.getResult().getType()) {
          unpacked = decoded.getResult();
        } else {
          // slice the tensor at the first N entries
          SmallVector<OpFoldResult> offsets({builder.getIndexAttr(0)});
          SmallVector<OpFoldResult> sizes(
              {builder.getIndexAttr(tensorTy.getNumElements())});
          SmallVector<OpFoldResult> strides({builder.getIndexAttr(1)});
          auto sliceOp = builder.create<tensor::ExtractSliceOp>(
              op.getLoc(), decoded.getResult(), offsets, sizes, strides);
          unpacked = sliceOp.getResult();
        }
      } else {
        auto zero = builder.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto extractOp =
            builder.create<tensor::ExtractOp>(op.getLoc(), decoded.getResult(),
                                              /*indices=*/zero.getResult());
        unpacked = extractOp.getResult();
      }

      decValuesToReturn.push_back(unpacked);
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
  RingAttr ringEnc = ringEncResult.value();
  RingAttr ringDec = ringDecResult.value();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  // We need one encryption function per argument and one decryption
  // function per return value. This is mainly to avoid complicated C++ codegen
  // when encrypting multiple inputs which requires out-params.
  for (BlockArgument val : op.getArguments()) {
    auto argTy = val.getType();
    if (auto argCtTy = dyn_cast<lwe::NewLWECiphertextType>(argTy)) {
      std::string encFuncName("");
      llvm::raw_string_ostream encNameOs(encFuncName);
      encNameOs << op.getSymName() << "__encrypt__arg" << val.getArgNumber();
      auto originalTypeAttr = op.getArgAttrOfType<OriginalTypeAttr>(
          val.getArgNumber(), kOriginalTypeAttrName);
      if (failed(generateEncryptionFunc(
              op, encFuncName, {originalTypeAttr.getOriginalType()}, {argCtTy},
              usePublicKey, ringEnc, {originalTypeAttr}, builder))) {
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
      auto originalTypeAttr =
          op.getResultAttrOfType<OriginalTypeAttr>(i, kOriginalTypeAttrName);
      if (failed(generateDecryptionFunc(op, decFuncName, {returnCtTy},
                                        {originalTypeAttr.getOriginalType()},
                                        ringDec, {originalTypeAttr},
                                        builder))) {
        return failure();
      }
      // insertion point is inside func, move back out
      builder.setInsertionPointToEnd(module.getBody());
    }
  }

  return success();
}

class LowerAssignLayout : public OpRewritePattern<tensor_ext::AssignLayoutOp> {
 public:
  LowerAssignLayout(mlir::MLIRContext *context, int64_t ciphertextSize)
      : OpRewritePattern<tensor_ext::AssignLayoutOp>(context),
        ciphertextSize(ciphertextSize) {}

  LogicalResult matchAndRewrite(tensor_ext::AssignLayoutOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(op);
    auto res = implementAssignLayout(op, ciphertextSize, b,
                                     [&](Operation *createdOp) {});
    if (failed(res)) return failure();
    rewriter.replaceOp(op, res.value());
    return success();
  };

 private:
  int64_t ciphertextSize;
};

struct AddClientInterface : impl::AddClientInterfaceBase<AddClientInterface> {
  using AddClientInterfaceBase::AddClientInterfaceBase;

  bool getUsePublicKey() {
    return llvm::TypeSwitch<Attribute, bool>(getSchemeParamAttr(getOperation()))
        .Case<bgv::SchemeParamAttr>([](auto attr) {
          return attr.getEncryptionType() == bgv::BGVEncryptionType::pk;
        })
        .Case<ckks::SchemeParamAttr>([](auto attr) {
          return attr.getEncryptionType() == ckks::CKKSEncryptionType::pk;
        })
        .Default([this](Attribute attr) {
          assert(false && "Unsupported scheme parame attribute");
          signalPassFailure();
          return false;
        });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *root = getOperation();

    bool usePk = getUsePublicKey();
    auto result = root->walk<WalkOrder::PreOrder>([&](func::FuncOp op) {
      if (op.isDeclaration()) {
        op->emitWarning("Skipping client interface for external func");
        return WalkResult::advance();
      }
      if (failed(convertFunc(op, usePk))) {
        op->emitError("Failed to add client interface for func");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();

    int ciphertextSize = getNumSlots(getOperation());
    LLVM_DEBUG(llvm::dbgs()
                   << "Detected num slots=" << ciphertextSize << "\n";);
    RewritePatternSet patterns(context);
    patterns.add<LowerAssignLayout>(context, ciphertextSize);
    walkAndApplyPatterns(root, std::move(patterns));
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
