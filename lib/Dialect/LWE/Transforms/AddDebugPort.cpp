#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"

#include <string>
#include <utility>

#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDDEBUGPORT
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

FailureOr<Type> getPrivateKeyType(func::FuncOp op) {
  SmallVector<Type, 4> ciphertextTypes;
  for (Type type : op.getArgumentTypes()) {
    Type elementType = getElementTypeOrSelf(type);
    if (isa<LWECiphertextType>(elementType)) {
      ciphertextTypes.push_back(elementType);
    }
  }

  if (ciphertextTypes.empty()) {
    return failure();
  }

  if (!llvm::all_equal(ciphertextTypes)) {
    op.emitWarning(
        "Conflicting ciphertext types found among function arguments");
  }

  // Fallback to the first ciphertext type. This is acceptable assuming that
  // all ciphertexts in the function are intended to be decrypted by the same
  // key, or at least that the key derived from the first ciphertext is
  // sufficient for debugging purposes.
  auto lweCiphertextType = cast<LWECiphertextType>(ciphertextTypes[0]);
  auto lwePrivateKeyType =
      LWESecretKeyType::get(op.getContext(), lweCiphertextType.getKey(),
                            lweCiphertextType.getCiphertextSpace().getRing());
  return lwePrivateKeyType;
}

func::FuncOp getOrCreateExternalDebugFunc(
    ModuleOp module, Type lwePrivateKeyType, Type valueType,
    llvm::DenseMap<std::pair<Type, Type>, int>& typePairToInt) {
  auto key = std::make_pair(lwePrivateKeyType, valueType);
  if (typePairToInt.find(key) == typePairToInt.end()) {
    typePairToInt[key] = typePairToInt.size();
  }

  auto* context = module.getContext();
  auto debugFuncType =
      FunctionType::get(context, {lwePrivateKeyType, valueType}, {});

  int counter = typePairToInt[key];
  std::string funcName = "__heir_debug_" + std::to_string(counter);

  while (auto lookup = module.lookupSymbol<func::FuncOp>(funcName)) {
    if (lookup.getFunctionType() == debugFuncType) return lookup;
    // Name conflict with different type, try next name
    funcName = "__heir_debug_" + std::to_string(++counter);
  }

  // Update the map with the actual counter used, to avoid searching again
  typePairToInt[key] = counter;

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  auto funcOp = func::FuncOp::create(b, funcName, debugFuncType);
  // required for external func call
  funcOp.setPrivate();
  return funcOp;
}

void insertValidationOps(func::FuncOp op) {
  int count = 0;
  auto insertValidate = [&](Value value, OpBuilder& b) {
    Type valueType = value.getType();
    if (isa<LWECiphertextType>(getElementTypeOrSelf(valueType))) {
      debug::ValidateOp::create(b, value.getLoc(), value,
                                "heir_debug_" + std::to_string(count++),
                                nullptr);
    }
  };

  Block& entryBlock = op.getBody().getBlocks().front();
  OpBuilder argBuilder(&entryBlock, entryBlock.begin());
  for (auto arg : op.getArguments()) {
    insertValidate(arg, argBuilder);
  }

  op.walk([&](Operation* walkOp) {
    if (walkOp == op.getOperation() || isa<tensor::EmptyOp>(walkOp) ||
        walkOp->hasTrait<OpTrait::IsTerminator>())
      return;
    OpBuilder opBuilder(walkOp->getBlock(), ++walkOp->getIterator());
    for (Value result : walkOp->getResults()) {
      insertValidate(result, opBuilder);
    }
  });
}

LogicalResult lowerValidationOps(
    func::FuncOp op, Value privateKey, int messageSize,
    llvm::DenseMap<std::pair<Type, Type>, int>& typePairToInt) {
  auto module = op->getParentOfType<ModuleOp>();
  Type lwePrivateKeyType = privateKey.getType();

  auto walkResult = op.walk([&](debug::ValidateOp validateOp) {
    Value value = validateOp.getInput();
    Type valueType = value.getType();
    if (isa<LWECiphertextType>(getElementTypeOrSelf(valueType))) {
      ImplicitLocOpBuilder b(validateOp.getLoc(), validateOp);
      SmallVector<NamedAttribute> attrs;
      // Transfer metadata from validateOp to CallOp
      attrs.push_back(b.getNamedAttr("debug.name", validateOp.getNameAttr()));
      if (validateOp.getMetadata()) {
        attrs.push_back(
            b.getNamedAttr("debug.metadata", validateOp.getMetadataAttr()));
      }
      attrs.push_back(b.getNamedAttr(
          "message.size", b.getStringAttr(std::to_string(messageSize))));

      auto debugFunc = getOrCreateExternalDebugFunc(module, lwePrivateKeyType,
                                                    valueType, typePairToInt);
      auto callOp =
          b.create<func::CallOp>(debugFunc, ArrayRef<Value>{privateKey, value});
      callOp->setDialectAttrs(attrs);

      validateOp.erase();
    } else {
      validateOp.emitError(
          "only LWECiphertextType is supported for debug.validate");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

struct AddDebugPort : impl::AddDebugPortBase<AddDebugPort> {
  using AddDebugPortBase::AddDebugPortBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    llvm::DenseMap<std::pair<Type, Type>, int> typePairToInt;
    llvm::DenseMap<func::FuncOp, Type> funcToKeyType;
    SmallVector<func::FuncOp, 16> worklist;

    SymbolTable symbolTable(module);
    llvm::DenseMap<func::FuncOp, SmallVector<func::CallOp, 4>> calleeToCalls;
    llvm::DenseSet<func::FuncOp> modifiedFuncs;

    if (failed(identifyInitialTargets(module, symbolTable, funcToKeyType,
                                      worklist))) {
      signalPassFailure();
      return;
    }

    if (failed(propagateKeyTypes(module, funcToKeyType, worklist,
                                 calleeToCalls))) {
      signalPassFailure();
      return;
    }

    if (insertDebugAfterEveryOp) {
      for (auto& [func, _] : funcToKeyType) {
        insertValidationOps(func);
      }
    }

    if (failed(addKeyArguments(funcToKeyType, modifiedFuncs))) {
      signalPassFailure();
      return;
    }

    if (failed(updateCallSites(modifiedFuncs, calleeToCalls, funcToKeyType))) {
      signalPassFailure();
      return;
    }

    if (failed(lowerAllValidationOps(module, funcToKeyType, typePairToInt))) {
      signalPassFailure();
      return;
    }
  }

 private:
  /// Step 1: Identify initial targets for debug port insertion.
  ///
  /// This function scans the module for functions that need to be processed.
  /// If an entry function is specified, it only processes that function.
  /// Otherwise, it processes functions that have validation ops or if the
  /// `insertDebugAfterEveryOp` flag is set, functions that have at least one
  /// LWE ciphertext argument.
  ///
  /// \param module The module to process.
  /// \param symbolTable The symbol table for the module.
  /// \param funcToKeyType Output map from function to its inferred key type.
  /// \param worklist Output list of functions to process.
  /// \return success() if successful, failure() otherwise.
  LogicalResult identifyInitialTargets(
      ModuleOp module, SymbolTable& symbolTable,
      llvm::DenseMap<func::FuncOp, Type>& funcToKeyType,
      SmallVector<func::FuncOp, 16>& worklist) {
    func::FuncOp entryFunc;
    if (!entryFunction.empty()) {
      entryFunc = symbolTable.lookup<func::FuncOp>(entryFunction);
    }

    if (entryFunc) {
      auto type = getPrivateKeyType(entryFunc);
      if (succeeded(type)) {
        funcToKeyType[entryFunc] = *type;
        worklist.push_back(entryFunc);
        return success();
      }

      if (containsAnyOperations<debug::ValidateOp>(entryFunc)) {
        entryFunc.emitError(
            "Cannot infer LWE private key type for entry function");
        return failure();
      }
    }

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isExternal()) continue;

      bool shouldProcess = containsAnyOperations<debug::ValidateOp>(funcOp);
      if (!shouldProcess && insertDebugAfterEveryOp) {
        shouldProcess = succeeded(getPrivateKeyType(funcOp));
      }

      if (shouldProcess) {
        auto type = getPrivateKeyType(funcOp);
        if (failed(type)) {
          return funcOp.emitError(
              "Cannot infer LWE private key type for function with "
              "validation ops");
        }

        funcToKeyType[funcOp] = *type;
        worklist.push_back(funcOp);
      }
    }
    return success();
  }

  /// Step 2: Propagate key types up the call graph.
  ///
  /// This function propagates the inferred key types from callees to callers.
  /// It also populates the `calleeToCalls` map to keep track of call sites.
  ///
  /// \param module The module to process.
  /// \param funcToKeyType Map from function to its inferred key type.
  /// \param worklist List of functions to process.
  /// \param calleeToCalls Output map from callee to its call sites.
  /// \return success() if successful, failure() otherwise.
  LogicalResult propagateKeyTypes(
      ModuleOp module, llvm::DenseMap<func::FuncOp, Type>& funcToKeyType,
      SmallVector<func::FuncOp, 16>& worklist,
      llvm::DenseMap<func::FuncOp, SmallVector<func::CallOp, 4>>&
          calleeToCalls) {
    while (!worklist.empty()) {
      func::FuncOp currentFunc = worklist.back();
      worklist.pop_back();
      Type keyType = funcToKeyType[currentFunc];

      auto symbolUses = SymbolTable::getSymbolUses(currentFunc, module);
      if (symbolUses) {
        for (auto use : *symbolUses) {
          Operation* user = use.getUser();
          auto callOp = dyn_cast<func::CallOp>(user);
          if (!callOp) continue;

          calleeToCalls[currentFunc].push_back(callOp);

          func::FuncOp caller = callOp->getParentOfType<func::FuncOp>();
          if (!caller) continue;

          if (funcToKeyType.find(caller) == funcToKeyType.end()) {
            funcToKeyType[caller] = keyType;
            worklist.push_back(caller);
          } else if (funcToKeyType[caller] != keyType) {
            caller.emitError("Conflicting LWE private key types required");
            return failure();
          }
        }
      }
    }
    return success();
  }

  /// Step 3: Add key arguments to functions.
  ///
  /// This function adds the LWE private key as the first argument to functions
  /// that need it and don't already have it.
  ///
  /// \param funcToKeyType Map from function to its inferred key type.
  /// \param modifiedFuncs Output set of functions that were modified.
  /// \return success() if successful, failure() otherwise.
  LogicalResult addKeyArguments(
      llvm::DenseMap<func::FuncOp, Type>& funcToKeyType,
      llvm::DenseSet<func::FuncOp>& modifiedFuncs) {
    for (auto& [funcOp, keyType] : funcToKeyType) {
      bool hasKey = llvm::any_of(funcOp.getArguments(), [&](Value arg) {
        return arg.getType() == keyType;
      });

      if (!hasKey) {
        if (failed(
                funcOp.insertArgument(0, keyType, nullptr, funcOp.getLoc()))) {
          funcOp.emitError("failed to insert private key argument");
          return failure();
        }
        modifiedFuncs.insert(funcOp);
      }
    }
    return success();
  }

  /// Step 4: Update call sites to pass the key argument.
  ///
  /// This function updates the call sites of modified functions to pass the
  /// required LWE private key argument.
  ///
  /// \param modifiedFuncs Set of functions that were modified by adding a key
  /// argument.
  /// \param calleeToCalls Map from callee to its call sites.
  /// \param funcToKeyType Map from function to its inferred key type.
  /// \return success() if successful, failure() otherwise.
  LogicalResult updateCallSites(
      const llvm::DenseSet<func::FuncOp>& modifiedFuncs,
      const llvm::DenseMap<func::FuncOp, SmallVector<func::CallOp, 4>>&
          calleeToCalls,
      const llvm::DenseMap<func::FuncOp, Type>& funcToKeyType) {
    for (auto funcOp : modifiedFuncs) {
      auto it = calleeToCalls.find(funcOp);
      if (it == calleeToCalls.end()) continue;

      for (auto callOp : it->second) {
        auto callerFunc = callOp->getParentOfType<func::FuncOp>();
        auto keyIt = funcToKeyType.find(funcOp);
        if (keyIt == funcToKeyType.end()) {
          return failure();
        }
        Type keyType = keyIt->second;

        auto* keyToPass =
            llvm::find_if(callerFunc.getArguments(),
                          [&](Value arg) { return arg.getType() == keyType; });

        if (keyToPass == callerFunc.getArguments().end()) {
          callOp.emitError(
              "Caller does not have the required LWE private key argument");
          return failure();
        }

        SmallVector<Value, 4> operands;
        operands.push_back(*keyToPass);
        operands.append(callOp.getOperands().begin(),
                        callOp.getOperands().end());

        OpBuilder b(callOp);
        auto newCall =
            func::CallOp::create(b, callOp.getLoc(), funcOp, operands);
        newCall->setDialectAttrs(callOp->getDialectAttrs());
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          callOp.getResult(i).replaceAllUsesWith(newCall.getResult(i));
        }
        callOp.erase();
      }
    }
    return success();
  }

  /// Step 5: Lower all validation ops to external calls.
  ///
  /// This function lowers all `debug.validate` ops in the module to calls to
  /// external debug functions.
  ///
  /// \param module The module to process.
  /// \param funcToKeyType Map from function to its inferred key type.
  /// \param typePairToInt Map to track generated debug function names.
  /// \return success() if successful, failure() otherwise.
  LogicalResult lowerAllValidationOps(
      ModuleOp module, const llvm::DenseMap<func::FuncOp, Type>& funcToKeyType,
      llvm::DenseMap<std::pair<Type, Type>, int>& typePairToInt) {
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isExternal()) continue;

      Type keyType;
      Value privateKey;
      auto it = funcToKeyType.find(funcOp);
      if (it != funcToKeyType.end()) {
        keyType = it->second;
        privateKey = *llvm::find_if(funcOp.getArguments(), [&](Value arg) {
          return arg.getType() == keyType;
        });
      } else {
        for (auto arg : funcOp.getArguments()) {
          if (isa<LWESecretKeyType>(arg.getType())) {
            keyType = arg.getType();
            privateKey = arg;
            break;
          }
        }
      }

      if (privateKey) {
        if (failed(lowerValidationOps(funcOp, privateKey, messageSize,
                                      typePairToInt))) {
          funcOp.emitError("failed to lower validation ops");
          return failure();
        }
      } else if (containsAnyOperations<debug::ValidateOp>(funcOp)) {
        funcOp.emitError(
            "validation operations cannot be lowered without a private key");
        return failure();
      }
    }
    return success();
  }
};

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
