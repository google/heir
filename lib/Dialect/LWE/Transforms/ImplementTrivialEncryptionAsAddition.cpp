#include "lib/Dialect/LWE/Transforms/ImplementTrivialEncryptionAsAddition.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

#include "lib/Dialect/FuncUtils.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Format.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "implement-trivial-encryption-as-addition"

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_IMPLEMENTTRIVIALENCRYPTIONASADDITION
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

using func::FuncOp;

using EncryptKey = std::pair<Type, Attribute>;
using IndexMap = DenseMap<EncryptKey, int>;
using FuncToIndexMap = DenseMap<func::FuncOp, IndexMap>;

bool detectPublicKeyFromClientHelpers(ModuleOp module) {
  auto result = module.walk([&](func::FuncOp func) {
    if (isClientHelper(func.getOperation())) {
      for (Type ty : func.getArgumentTypes())
        if (isa<LWEPublicKeyType>(ty)) {
          return WalkResult::interrupt();
        }
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

// Creates a function that returns a single ciphertext encrypting zero. A new
// function is created for each ciphertext type returned by originalOp and each
// mgmt attribute attached to the originalOp, and otherwise duplicate functions
// are looked up by symbol name. Created functions are tagged with
// client.enc_zero_func.
func::FuncOp getOrCreateEncryptionOfZerosFunc(func::FuncOp parentFunc,
                                              TrivialEncryptOp originalOp,
                                              ModuleOp module, int index) {
  LWEPlaintextType plaintextType = cast<LWEPlaintextType>(
      getElementTypeOrSelf(originalOp.getInput().getType()));
  LWECiphertextType ciphertextType = cast<LWECiphertextType>(
      getElementTypeOrSelf(originalOp.getResult().getType()));

  SmallString<16> buffer;
  llvm::raw_svector_ostream os(buffer);
  os << parentFunc.getSymName() << "__encrypt__zero__" << index;
  SmallString<16> buffer2;
  std::string encFuncName = std::string(sanitizeIdentifier(buffer, buffer2));

  if (auto existingFunc = module.lookupSymbol<func::FuncOp>(encFuncName)) {
    assert(existingFunc->hasAttr(kClientEncZeroFuncAttrName) &&
           "existing function does not have client.enc_zero_func attribute");
    return existingFunc;
  }

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());
  builder.setInsertionPointAfter(parentFunc);

  bool usePublicKey = detectPublicKeyFromClientHelpers(module);
  Type keyTy;
  if (usePublicKey) {
    keyTy = lwe::LWEPublicKeyType::get(
        module.getContext(), lwe::KeyAttr::get(module.getContext(), 0),
        ciphertextType.getCiphertextSpace().getRing());
  } else {
    keyTy = lwe::LWESecretKeyType::get(
        module.getContext(), lwe::KeyAttr::get(module.getContext(), 0),
        ciphertextType.getCiphertextSpace().getRing());
  }

  FunctionType encFuncType =
      FunctionType::get(builder.getContext(), {keyTy}, {ciphertextType});
  auto encFuncOp = func::FuncOp::create(builder, encFuncName, encFuncType);

  encFuncOp->setAttr(kClientEncZeroFuncAttrName, builder.getUnitAttr());
  Block* entryBlock = encFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);

  int numSlots;
  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      module->getAttr(kRequestedSlotCountAttrName));
  if (numSlotsAttr) {
    numSlots = numSlotsAttr.getInt();
  } else {
    module->emitWarning()
        << "Encountered module op with no requested_slots "
           "attribute; defaulting to using polynomial modulus ring\n";
    numSlots = plaintextType.getPlaintextSpace()
                   .getRing()
                   .getPolynomialModulus()
                   .getPolynomial()
                   .getDegree();
    if (moduleIsCKKS(module)) {
      numSlots /= 2;
    }
  }

  Type coeffType =
      plaintextType.getPlaintextSpace().getRing().getCoefficientType();
  Type cleartextElementType =
      llvm::TypeSwitch<Type, Type>(coeffType)
          .Case<IntegerType, FloatType>([&](auto ty) { return ty; })
          .Case<mod_arith::ModArithType>(
              [&](auto ty) { return ty.getModulus().getType(); })
          .Default([&](auto ty) {
            originalOp->emitOpError()
                << "has unsupported plaintext coefficient type; can't "
                   "determine what constant type to use for encryption of "
                   "zero.";
            return Type();
          });
  if (!cleartextElementType) return nullptr;

  RankedTensorType zeroType =
      RankedTensorType::get({numSlots}, cleartextElementType);
  auto zeroAttr = builder.getZeroAttr(zeroType);
  arith::ConstantOp constantOp = arith::ConstantOp::create(builder, zeroAttr);

  auto plaintextSpace = plaintextType.getPlaintextSpace();
  auto encodeOp = RLWEEncodeOp::create(
      builder, plaintextType, constantOp.getResult(),
      plaintextSpace.getEncoding(), plaintextSpace.getRing());
  auto encrypted = RLWEEncryptOp::create(
      builder, ciphertextType, encodeOp.getResult(), encFuncOp.getArgument(0));
  encrypted->setAttrs(originalOp->getAttrs());

  func::ReturnOp::create(builder, encrypted.getResult());
  return encFuncOp;
}

// Creates a new function arg containing a ciphertext encrypting zero
// with the attribute client.enc_zero_arg
Value getOrCreateNewFuncArg(func::FuncOp func, LWECiphertextType type,
                            PatternRewriter& rewriter) {
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (func.getArgument(i).getType() == type &&
        func.getArgAttr(i, kClientEncZeroArgAttrName)) {
      return func.getArgument(i);
    }
  }
  auto context = func.getContext();
  auto oldType = func.getFunctionType();
  SmallVector<Type> newInputs(oldType.getInputs().begin(),
                              oldType.getInputs().end());
  newInputs.push_back(type);
  auto newType = FunctionType::get(context, newInputs, oldType.getResults());
  func.setType(newType);

  auto newArg = func.getBody().addArgument(type, func.getLoc());
  func.setArgAttr(newArg.getArgNumber(), kClientEncZeroArgAttrName,
                  rewriter.getUnitAttr());
  return newArg;
}

struct TrivialEncryptionRewritePattern
    : public OpRewritePattern<TrivialEncryptOp> {
  TrivialEncryptionRewritePattern(MLIRContext* context,
                                  const FuncToIndexMap& funcToIndexMap)
      : OpRewritePattern<TrivialEncryptOp>(context),
        funcToIndexMap(funcToIndexMap) {}

  LogicalResult matchAndRewrite(TrivialEncryptOp op,
                                PatternRewriter& rewriter) const override {
    auto func = op->getParentOfType<FuncOp>();
    assert(func && "TrivialEncryptOp must be nested within a FuncOp");
    auto module = func->getParentOfType<ModuleOp>();

    Type resultType = op.getResult().getType();
    LWECiphertextType ctTy =
        cast<LWECiphertextType>(getElementTypeOrSelf(resultType));
    Attribute mgmtAttr = op->getAttr(mgmt::MgmtDialect::kArgMgmtAttrName);
    EncryptKey key = {ctTy, mgmtAttr};

    auto funcIt = funcToIndexMap.find(func);
    assert(funcIt != funcToIndexMap.end() && "parent func not found in map");
    auto indexIt = funcIt->second.find(key);
    assert(indexIt != funcIt->second.end() && "encrypt key not found in map");
    int index = indexIt->second;

    getOrCreateEncryptionOfZerosFunc(func, op, module, index);
    Value newFuncArg = getOrCreateNewFuncArg(func, ctTy, rewriter);

    // The newFuncArg is a single ciphertext, so we may need to splat it
    // into a tensor of the appropriate shape
    Value operand = newFuncArg;
    if (isa<ShapedType>(op.getResult().getType())) {
      operand =
          tensor::SplatOp::create(rewriter, op.getLoc(), resultType, operand);
    }
    auto newAddOp =
        RAddPlainOp::create(rewriter, op.getLoc(), operand, op.getInput());
    rewriter.replaceOp(op, newAddOp);
    return success();
  }

 private:
  const FuncToIndexMap& funcToIndexMap;
};

struct ImplementTrivialEncryptionAsAddition
    : impl::ImplementTrivialEncryptionAsAdditionBase<
          ImplementTrivialEncryptionAsAddition> {
  using ImplementTrivialEncryptionAsAdditionBase::
      ImplementTrivialEncryptionAsAdditionBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ModuleOp module = cast<ModuleOp>(getOperation());

    FuncToIndexMap funcToIndexMap;
    module.walk([&](TrivialEncryptOp op) {
      auto func = op->getParentOfType<FuncOp>();
      assert(func && "TrivialEncryptOp must be nested within a FuncOp");
      LWECiphertextType ctTy = cast<LWECiphertextType>(
          getElementTypeOrSelf(op.getResult().getType()));
      Attribute mgmtAttr = op->getAttr(mgmt::MgmtDialect::kArgMgmtAttrName);
      EncryptKey key = {ctTy, mgmtAttr};

      auto& indexMap = funcToIndexMap[func];
      indexMap.insert({key, (int)indexMap.size()});
    });

    RewritePatternSet patterns(context);
    patterns.add<TrivialEncryptionRewritePattern>(context, funcToIndexMap);
    walkAndApplyPatterns(module, std::move(patterns));
  }
};

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
