#include "lib/Transforms/GatherZeroEncryptions/GatherZeroEncryptions.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/BitVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GATHERZEROENCRYPTIONS
#include "lib/Transforms/GatherZeroEncryptions/GatherZeroEncryptions.h.inc"

namespace {

struct ZeroArgInfo {
  // The index of the zero encryption in the new memref
  int64_t zeroEncIndex;

  // The original argument index.
  unsigned originalArgNum;

  // A reference to the block argument of the zero encryption before
  // it is erased.
  BlockArgument arg;
};

struct GatheredFuncInfo {
  // The func with zero encryption arguments
  func::FuncOp func;

  // A list of zero args discovered
  SmallVector<ZeroArgInfo> zeroArgs;

  int minZeroArgIndex() const {
    return *llvm::min_element(llvm::map_range(
        zeroArgs,
        [](const ZeroArgInfo& z) -> int { return z.originalArgNum; }));
  }
};

}  // namespace

struct GatherZeroEncryptions
    : impl::GatherZeroEncryptionsBase<GatherZeroEncryptions> {
  using GatherZeroEncryptionsBase::GatherZeroEncryptionsBase;

  // Collect the arguments of a function marked with as client-provided
  // encryptions of zero.
  SmallVector<ZeroArgInfo> getZeroArgs(func::FuncOp func) {
    SmallVector<ZeroArgInfo> zeroArgs;
    for (BlockArgument arg : func.getArguments()) {
      unsigned argNum = arg.getArgNumber();
      if (auto dictAttr = func.getArgAttrOfType<DictionaryAttr>(
              argNum, kClientEncZeroArgAttrName)) {
        int64_t index = 0;
        if (auto idxAttr = dictAttr.getAs<IntegerAttr>(kClientHelperIndex)) {
          index = idxAttr.getInt();
        }
        zeroArgs.push_back({index, argNum, arg});
      }
    }
    return zeroArgs;
  }

  // Gather the functions this pass should process. These are
  // all functions that may have an encryption of zero needed as
  // an input. This also summarizes relevant information about
  // each func for the rest of the pass to use.
  DenseMap<func::FuncOp, GatheredFuncInfo> gatherFuncs(ModuleOp module) {
    DenseMap<func::FuncOp, GatheredFuncInfo> gatheredFuncs;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration() || hasInterfaceRole(func, kClientEncZeroRole)) {
        continue;
      }

      SmallVector<ZeroArgInfo> zeroArgs = getZeroArgs(func);
      if (zeroArgs.empty()) {
        continue;
      }

      // These should already be sorted in the input function, so this sort is
      // mainly defensive.
      llvm::sort(zeroArgs, [](const ZeroArgInfo& a, const ZeroArgInfo& b) {
        return a.zeroEncIndex < b.zeroEncIndex;
      });

      gatheredFuncs.insert({func, {.func = func, .zeroArgs = zeroArgs}});
    }
    return gatheredFuncs;
  }

  // Update the function type for the merged function
  void updateFunctionType(func::FuncOp func, MemRefType memrefType,
                          int newArgIndex) {
    SmallVector<Type> newArgTypes;
    SmallVector<DictionaryAttr> newArgAttrs;
    auto* ctx = func.getContext();

    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      bool isZeroArg = func.getArgAttr(i, kClientEncZeroArgAttrName) != nullptr;
      if (!isZeroArg) {
        newArgTypes.push_back(func.getArgument(i).getType());
        DictionaryAttr oldDict = func.getArgAttrDict(i);
        newArgAttrs.push_back(oldDict ? oldDict : DictionaryAttr::get(ctx, {}));
      } else if (i == newArgIndex) {
        newArgTypes.push_back(memrefType);
        // The new argument now stands alone, and we don't need to keep track
        // of any extra information about it. We leave a unit attr to mark it
        // as holding encryptions of zero mainly for readability of the
        // generated MLIR.
        NamedAttribute unitAttr(StringAttr::get(ctx, kClientEncZeroArgAttrName),
                                UnitAttr::get(ctx));
        newArgAttrs.push_back(DictionaryAttr::get(ctx, unitAttr));
      }
    }

    func.setType(FunctionType::get(ctx, newArgTypes, func.getResultTypes()));
    func.setAllArgAttrs(newArgAttrs);
  }

  FailureOr<BlockArgument> getZeroEncMemrefArg(func::FuncOp func) {
    for (auto arg : func.getArguments()) {
      if (isa<MemRefType>(arg.getType()) &&
          func.getArgAttrOfType<UnitAttr>(arg.getArgNumber(),
                                          kClientEncZeroArgAttrName)) {
        return arg;
        break;
      }
    }
    return failure();
  }

  LogicalResult mergeZeroEncArgsAndUpdateFuncBody(GatheredFuncInfo info) {
    func::FuncOp func = info.func;
    // Assert all the zero encryption args have the same type, otherwise
    // they cannot be part of the same memref. This pass must not be invoked
    // at the LWE dialect level, since there ciphertext types are not
    // opaque. Instead it should be invoked at backends like lattigo and
    // openfhe that have opaque ciphertext types.
    if (!llvm::all_equal(llvm::map_range(
            info.zeroArgs,
            [](ZeroArgInfo argInfo) { return argInfo.arg.getType(); }))) {
      info.func.emitOpError()
          << "must have zero encryption args with identical types";
      return failure();
    }

    int64_t n = info.zeroArgs.size();
    Type ctTy = info.zeroArgs[0].arg.getType();
    MemRefType memrefType = MemRefType::get({n}, ctTy);
    int firstZeroArgIdx = info.minZeroArgIndex();

    updateFunctionType(func, memrefType,
                       /*newArgIndex=*/firstZeroArgIdx);

    Block* entryBlock = &func.getBody().front();
    BlockArgument newMemrefArg =
        entryBlock->insertArgument(firstZeroArgIdx, memrefType, func.getLoc());

    // Update internal references to a soon-to-be-erased encryption-of-zero
    // block arg to the corresponding memref load of the new memref arg.
    OpBuilder builder(entryBlock, entryBlock->begin());
    for (int64_t k = 0; k < n; ++k) {
      Value idxVal = arith::ConstantIndexOp::create(builder, func.getLoc(), k);
      Value loaded = memref::LoadOp::create(builder, func.getLoc(),
                                            newMemrefArg, ValueRange{idxVal});
      info.zeroArgs[k].arg.replaceAllUsesWith(loaded);
    }

    // Erase all the now-unused block args. Use a BitVector and
    // Block::eraseArguments to properly handle index shifting as arguments are
    // erased.
    llvm::BitVector indicesToErase(entryBlock->getNumArguments());
    for (const ZeroArgInfo& argInfo : info.zeroArgs) {
      indicesToErase.set(argInfo.arg.getArgNumber());
    }
    entryBlock->eraseArguments(indicesToErase);

    return success();
  }

  LogicalResult updateCallOp(
      func::CallOp callOp, ModuleOp module,
      DenseMap<func::FuncOp, GatheredFuncInfo>& gatheredFuncs) {
    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    const GatheredFuncInfo& calleeInfo = gatheredFuncs[callee];
    auto caller = callOp->getParentOfType<func::FuncOp>();

    FailureOr<BlockArgument> maybeMemrefToPass = getZeroEncMemrefArg(caller);
    if (failed(maybeMemrefToPass)) {
      caller.emitOpError()
          << " contains call to function accepting zero-encryptions, but "
             "contains no zero encryption memref argument to pass along";
      return failure();
    }
    Value memrefToPass = *maybeMemrefToPass;

    OpBuilder builder(callOp);
    SmallVector<Value> newOperands;
    int firstZeroArgIdx = calleeInfo.minZeroArgIndex();
    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      bool isZeroArg = llvm::any_of(
          calleeInfo.zeroArgs,
          [&](const ZeroArgInfo& z) { return z.originalArgNum == i; });
      if (!isZeroArg) {
        newOperands.push_back(callOp.getOperand(i));
      } else if (i == firstZeroArgIdx) {
        newOperands.push_back(memrefToPass);
      }
    }

    auto newCallOp =
        func::CallOp::create(builder, callOp.getLoc(), callOp.getCallee(),
                             callOp.getResultTypes(), newOperands);
    callOp.replaceAllUsesWith(newCallOp);
    callOp.erase();

    return success();
  }

  // Gets the target function for a zero-encryption func. Returns failure
  // if no target function name could be found, or the func is not a zero
  // encryption helper.
  FailureOr<StringRef> getTargetFuncForZeroEncFunc(func::FuncOp func) {
    if (auto dict = getInterfaceAttr(func, kClientEncZeroRole)) {
      if (auto funcNameAttr = dict.getAs<StringAttr>(kClientHelperFuncName)) {
        return funcNameAttr.getValue();
      }
    }

    // If the attribute is not found or malformed, we could fall back to
    // string munging the symbol name of the func, but we would prefer to
    // eagerly fail and report an error.
    if (hasInterfaceRole(func, kClientEncZeroRole)) {
      func.emitOpError() << " expected to contain attr naming the func this "
                            "zero-encryption helper targets.";
      signalPassFailure();
    }
    return failure();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1: update func signatures to use a memref type, and function bodies
    // to load from the memref.
    DenseMap<func::FuncOp, GatheredFuncInfo> gatheredFuncs =
        gatherFuncs(module);
    for (const GatheredFuncInfo& info : gatheredFuncs.values()) {
      if (failed(mergeZeroEncArgsAndUpdateFuncBody(info))) {
        signalPassFailure();
        return;
      }
    }

    // Step 2: Update internal func.call ops (e.g. if @main calls
    // @main__preprocessed) to propagate the new memref arg
    SmallVector<func::CallOp> callsToUpdate;
    module.walk([&](func::CallOp callOp) {
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (callee && gatheredFuncs.count(callee)) {
        callsToUpdate.push_back(callOp);
      }
    });

    for (func::CallOp callOp : callsToUpdate) {
      if (failed(updateCallOp(callOp, module, gatheredFuncs))) {
        signalPassFailure();
        return;
      }
    }

    // Step 3: group client helpers by target function, create a single
    // combined helper @<func>__encrypt__zeros(...) -> memref<N x !ct>.
    DenseMap<StringRef, SmallVector<func::FuncOp>> helpersByTarget;
    module.walk([&](func::FuncOp func) {
      auto maybeTarget = getTargetFuncForZeroEncFunc(func);
      if (failed(maybeTarget)) return;
      helpersByTarget[*maybeTarget].push_back(func);
    });

    for (auto& [targetFunc, helpers] : helpersByTarget) {
      auto getHelperIndex = [](func::FuncOp func) -> int64_t {
        if (auto dict = getInterfaceAttr(func, kClientEncZeroRole)) {
          if (auto idxAttr = dict.getAs<IntegerAttr>(kClientHelperIndex)) {
            return idxAttr.getInt();
          }
        }
        return 0;
      };

      llvm::sort(helpers, [&](func::FuncOp a, func::FuncOp b) {
        return getHelperIndex(a) < getHelperIndex(b);
      });

      int64_t n = helpers.size();
      Type ctTy = helpers[0].getResultTypes()[0];
      MemRefType memrefType = MemRefType::get({n}, ctTy);

      std::string combinedName = (targetFunc + "__encrypt__zeros").str();
      FunctionType funcType = FunctionType::get(
          memrefType.getContext(), helpers[0].getArgumentTypes(), {memrefType});

      OpBuilder builder(module.getContext());
      builder.setInsertionPointAfter(helpers.back());
      auto combinedFunc = func::FuncOp::create(builder, helpers[0].getLoc(),
                                               combinedName, funcType);
      combinedFunc.setVisibility(helpers[0].getVisibility());

      setInterfaceRole(
          combinedFunc, kClientEncZeroRole,
          builder.getDictionaryAttr({
              builder.getNamedAttr(kClientHelperFuncName,
                                   builder.getStringAttr(targetFunc)),
          }));

      for (unsigned i = 0; i < helpers[0].getNumArguments(); ++i) {
        if (auto attrs = helpers[0].getArgAttrDict(i)) {
          combinedFunc.setArgAttrs(i, attrs);
        }
      }

      Block* entryBlock = combinedFunc.addEntryBlock();
      builder.setInsertionPointToEnd(entryBlock);

      Location loc = combinedFunc.getLoc();
      Value allocated = memref::AllocOp::create(builder, loc, memrefType);

      for (auto [k, helper] : llvm::enumerate(helpers)) {
        IRMapping map;
        for (unsigned argIdx = 0; argIdx < helper.getNumArguments(); ++argIdx) {
          map.map(helper.getArgument(argIdx), combinedFunc.getArgument(argIdx));
        }

        for (auto& op : helper.getBody().front()) {
          if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
            Value retVal = map.lookup(returnOp.getOperand(0));
            Value idxVal = arith::ConstantIndexOp::create(builder, loc, k);
            memref::StoreOp::create(builder, loc, retVal, allocated,
                                    ValueRange{idxVal});
          } else {
            builder.clone(op, map);
          }
        }
      }

      func::ReturnOp::create(builder, loc, ValueRange{allocated});

      for (auto helper : helpers) {
        helper.erase();
      }
    }
  }
};

}  // namespace heir
}  // namespace mlir
