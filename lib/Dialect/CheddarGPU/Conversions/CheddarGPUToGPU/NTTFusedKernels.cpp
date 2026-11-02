#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/NTTFusedKernels.h"

#include <cstdint>
#include <string>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

void buildNTTFwdP1Body(ImplicitLocOpBuilder& ib, Block* entryBlock, Value smem,
                       unsigned W, size_t totalLimbs, bool mont,
                       bool hasSkipCopy) {
  Value outArg = entryBlock->getArgument(0);
  Value inArg = entryBlock->getArgument(1);
  Value twArg = entryBlock->getArgument(2);
  int argIdx = 3;
  Value r2Arg = mont ? entryBlock->getArgument(argIdx++) : Value();
  Value skipSrcArg = hasSkipCopy ? entryBlock->getArgument(argIdx++) : Value();
  Value pIndicesArg = entryBlock->getArgument(argIdx++);
  Value oStartArg = entryBlock->getArgument(argIdx++);
  Value iStartArg = entryBlock->getArgument(argIdx++);
  Value pOffsetArg = entryBlock->getArgument(argIdx++);
  Value sStartArg = entryBlock->getArgument(argIdx++);
  Value sEndArg = entryBlock->getArgument(argIdx++);
  Value skipInOffArg =
      hasSkipCopy ? entryBlock->getArgument(argIdx++) : Value();
  Value hIndicesArg = entryBlock->getArgument(argIdx++);

  Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
  Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
  Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);

  Value localLimb = by;
  Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);
  Value inLimb = ib.create<arith::AddIOp>(localLimb, iStartArg);

  Value ge =
      ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge, outLimb, sStartArg);
  Value lt =
      ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt, outLimb, sEndArg);
  Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
  Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
  Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);

  auto ifOp = ib.create<scf::IfOp>(notSkipped, /*withElseRegion=*/hasSkipCopy);
  ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

  Value c0 = ib.create<arith::ConstantIndexOp>(0);
  Value primeOffsetLimb = ib.create<arith::AddIOp>(localLimb, pOffsetArg);
  Value safeIdx = ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
  Value primeFromIdx =
      ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
  Value absLimb =
      ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx, primeOffsetLimb);

  auto [primesMemref, qInvMemref] = getDevicePrimeGlobals(ib, W, totalLimbs);
  Value q = ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
  Value qInv = ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

  auto getTwiddle = [&](Value twIdxVal) -> Value {
    return ib.create<memref::LoadOp>(twArg, ValueRange{absLimb, twIdxVal});
  };

  Value c4 = ib.create<arith::ConstantIndexOp>(4);
  Value c8 = ib.create<arith::ConstantIndexOp>(8);
  Value c9 = ib.create<arith::ConstantIndexOp>(9);
  Value c13 = ib.create<arith::ConstantIndexOp>(13);
  Value c15 = ib.create<arith::ConstantIndexOp>(15);

  Value stageGroupIdx = ib.create<arith::ShRUIOp>(tx, c4);
  Value batchIdx = ib.create<arith::AndIOp>(tx, c15);

  Value bxShL4 = ib.create<arith::ShLIOp>(bx, c4);
  Value sgShL9 = ib.create<arith::ShLIOp>(stageGroupIdx, c9);
  Value loadBase = ib.create<arith::AddIOp>(batchIdx, bxShL4);
  loadBase = ib.create<arith::AddIOp>(loadBase, sgShL9);

  Value local[16];
  for (int i = 0; i < 16; ++i) {
    Value offset = ib.create<arith::ConstantIndexOp>(i << 12);
    Value srcIdx = ib.create<arith::AddIOp>(loadBase, offset);
    local[i] = ib.create<memref::LoadOp>(inArg, ValueRange{inLimb, srcIdx});
  }

  if (mont) {
    Value r2Val = ib.create<memref::LoadOp>(r2Arg, ValueRange{absLimb});
    for (int i = 0; i < 16; ++i) {
      local[i] = buildMontMulRuntime(ib, local[i], r2Val, q, qInv);
    }
  }

  // MultiRadixNTTFirst<word, 16, 3>(local, tw_idx = 1)
  // Stage 1
  {
    Value w = getTwiddle(ib.create<arith::ConstantIndexOp>(1));
    for (int j = 0; j < 8; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 8], w, q, qInv);
    }
  }
  // Stage 2
  {
    Value w0 = getTwiddle(ib.create<arith::ConstantIndexOp>(2));
    Value w1 = getTwiddle(ib.create<arith::ConstantIndexOp>(3));
    for (int j = 0; j < 4; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 4], w0, q, qInv);
      butterflyNTTRuntime(ib, local[8 + j], local[12 + j], w1, q, qInv);
    }
  }
  // Stage 3
  {
    for (int k = 0; k < 4; ++k) {
      Value wk = getTwiddle(ib.create<arith::ConstantIndexOp>(4 + k));
      for (int j = 0; j < 2; ++j) {
        butterflyNTTRuntime(ib, local[4 * k + j], local[4 * k + 2 + j], wk, q,
                            qInv);
      }
    }
  }

  // Store to shared memory: temp[tx + (j << 7)] = local[j]
  for (int j = 0; j < 16; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j << 7);
    Value smIdx = ib.create<arith::AddIOp>(tx, smOffset);
    ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
  }
  ib.create<gpu::BarrierOp>();

  // Transposed reload: sm_idx = (stage_group_idx << 8) + batch_idx
  Value sgShL8 = ib.create<arith::ShLIOp>(stageGroupIdx, c8);
  Value smLoadBase = ib.create<arith::AddIOp>(sgShL8, batchIdx);
  for (int j = 0; j < 16; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j << 4);
    Value smIdx = ib.create<arith::AddIOp>(smLoadBase, smOffset);
    local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
  }

  // MultiRadixNTT<word, 16, 4>(local, base_tw = 8 + stage_group_idx)
  Value c8Const = ib.create<arith::ConstantIndexOp>(8);
  Value baseTw = ib.create<arith::AddIOp>(c8Const, stageGroupIdx);

  // Stage 4
  {
    Value w = getTwiddle(baseTw);
    for (int j = 0; j < 8; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 8], w, q, qInv);
    }
  }
  // Stage 5
  {
    Value c1Const = ib.create<arith::ConstantIndexOp>(1);
    Value baseTwShL1 = ib.create<arith::ShLIOp>(baseTw, c1Const);
    Value w0 = getTwiddle(baseTwShL1);
    Value w1Idx = ib.create<arith::AddIOp>(baseTwShL1, c1Const);
    Value w1 = getTwiddle(w1Idx);
    for (int j = 0; j < 4; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 4], w0, q, qInv);
      butterflyNTTRuntime(ib, local[8 + j], local[12 + j], w1, q, qInv);
    }
  }
  // Stage 6
  {
    Value c2Const = ib.create<arith::ConstantIndexOp>(2);
    Value baseTwShL2 = ib.create<arith::ShLIOp>(baseTw, c2Const);
    for (int k = 0; k < 4; ++k) {
      Value kConst = ib.create<arith::ConstantIndexOp>(k);
      Value wkIdx = ib.create<arith::AddIOp>(baseTwShL2, kConst);
      Value wk = getTwiddle(wkIdx);
      for (int j = 0; j < 2; ++j) {
        butterflyNTTRuntime(ib, local[4 * k + j], local[4 * k + 2 + j], wk, q,
                            qInv);
      }
    }
  }
  // Stage 7
  {
    Value c3Const = ib.create<arith::ConstantIndexOp>(3);
    Value baseTwShL3 = ib.create<arith::ShLIOp>(baseTw, c3Const);
    for (int k = 0; k < 8; ++k) {
      Value kConst = ib.create<arith::ConstantIndexOp>(k);
      Value wkIdx = ib.create<arith::AddIOp>(baseTwShL3, kConst);
      Value wk = getTwiddle(wkIdx);
      butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q, qInv);
    }
  }

  // Store to op.getOut()
  Value sgShL13 = ib.create<arith::ShLIOp>(stageGroupIdx, c13);
  Value dstBase = ib.create<arith::AddIOp>(batchIdx, sgShL13);
  dstBase = ib.create<arith::AddIOp>(dstBase, bxShL4);
  for (int i = 0; i < 16; ++i) {
    Value offset = ib.create<arith::ConstantIndexOp>(i << 9);
    Value dstIdx = ib.create<arith::AddIOp>(dstBase, offset);
    ib.create<memref::StoreOp>(local[i], outArg, ValueRange{outLimb, dstIdx});
  }

  if (hasSkipCopy) {
    ib.setInsertionPointToStart(&ifOp.getElseRegion().front());
    Value outSubStart = ib.create<arith::SubIOp>(outLimb, sStartArg);
    Value skipLimb = ib.create<arith::AddIOp>(outSubStart, skipInOffArg);
    Value c2048 = ib.create<arith::ConstantIndexOp>(2048);
    Value bx2048 = ib.create<arith::MulIOp>(bx, c2048);
    Value baseCoeff = ib.create<arith::AddIOp>(bx2048, tx);
    for (int k = 0; k < 16; ++k) {
      Value kOffset = ib.create<arith::ConstantIndexOp>(k * 128);
      Value coeffIdx = ib.create<arith::AddIOp>(baseCoeff, kOffset);
      Value val =
          ib.create<memref::LoadOp>(skipSrcArg, ValueRange{skipLimb, coeffIdx});
      ib.create<memref::StoreOp>(val, outArg, ValueRange{outLimb, coeffIdx});
    }
  }

  ib.setInsertionPointAfter(ifOp);
  ib.create<gpu::ReturnOp>();
}

template <unsigned W>
void rewriteNTTFused(NTTOp op, PatternRewriter& rewriter, ParamsAttr paramsAttr,
                     int64_t inStart, int64_t outStart, int64_t primeStart,
                     int64_t lCount) {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  MLIRContext* context = b.getContext();
  Location loc = op.getLoc();

  Type intType = IntegerType::get(context, W);
  Type indexType = b.getIndexType();
  Type i1Type = b.getI1Type();

  int64_t degree = 1LL << paramsAttr.getLogN();
  size_t totalLimbs = paramsAttr.getNumLimbs();

  MemRefType dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, intType);
  MemRefType dyn1DMemrefType = MemRefType::get({ShapedType::kDynamic}, intType);
  MemRefType dyn1DIndexMemrefType =
      MemRefType::get({ShapedType::kDynamic}, indexType);

  getOrCreateModuleTwiddleGlobals<W>(module, paramsAttr, degree);

  std::string fwdName =
      llvm::formatv("__cheddar_twiddle_fwd_w{0}_deg{1}", W, degree).str();
  auto fwdGlobal = module.lookupSymbol<memref::GlobalOp>(fwdName);
  Value twiddlesVal =
      getOrCreateHostGetGlobal(b, loc, fwdGlobal, op.getOperation());
  Value twiddlesDyn = castToDynamicFirstDim(b, loc, twiddlesVal);

  int64_t skipStart = 0;
  int64_t skipEnd = 0;
  if (op.getSkipRange() && op.getSkipRange()->size() == 2) {
    skipStart = (*op.getSkipRange())[0];
    skipEnd = (*op.getSkipRange())[1];
  }

  memref::GlobalOp primeIndicesGlobal;
  bool hasPrimeIndices = false;
  int64_t primeOffset = primeStart;
  if (op.getPrimeIndices()) {
    primeIndicesGlobal =
        getOrCreatePrimeIndicesGlobal(module, *op.getPrimeIndices());
    hasPrimeIndices = true;
    primeOffset = 0;
  } else {
    primeIndicesGlobal =
        module.lookupSymbol<memref::GlobalOp>("__cheddar_dummy_indices");
  }
  Value primeIndicesVal =
      getOrCreateHostGetGlobal(b, loc, primeIndicesGlobal, op.getOperation());
  Value primeIndicesDyn = castToDynamic1D(b, loc, primeIndicesVal);

  Value outDyn = castToDynamicFirstDim(b, loc, op.getOut());
  Value inDyn = castToDynamicFirstDim(b, loc, op.getIn());
  Value outStartVal = b.create<arith::ConstantIndexOp>(loc, outStart);
  Value inStartVal = b.create<arith::ConstantIndexOp>(loc, inStart);
  Value primeOffsetVal = b.create<arith::ConstantIndexOp>(loc, primeOffset);
  Value skipStartVal = b.create<arith::ConstantIndexOp>(loc, skipStart);
  Value skipEndVal = b.create<arith::ConstantIndexOp>(loc, skipEnd);
  Value hasPrimeIndicesVal =
      b.create<arith::ConstantIntOp>(loc, hasPrimeIndices ? 1 : 0, 1);

  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value cLCount = b.create<arith::ConstantIndexOp>(loc, lCount);
  Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
  Value c64 = b.create<arith::ConstantIndexOp>(loc, 64);
  Value c128 = b.create<arith::ConstantIndexOp>(loc, 128);

  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  // === Phase 1 Launch ===
  gpu::KernelDim3 gridP1{c32, cLCount, c1};
  gpu::KernelDim3 blockP1{c128, c1, c1};

  bool mont = op.getMontgomeryConversion();
  bool hasSkipCopy = (op.getSkipCopySrc() != nullptr);
  std::string p1KernelName;
  if (hasSkipCopy) {
    p1KernelName =
        mont
            ? llvm::formatv(
                  "cheddar_gpu_ntt_fused_fwd_p1_mont_skipcopy_w{0}_deg{1}", W,
                  degree)
                  .str()
            : llvm::formatv("cheddar_gpu_ntt_fused_fwd_p1_skipcopy_w{0}_deg{1}",
                            W, degree)
                  .str();
  } else {
    p1KernelName =
        mont ? llvm::formatv("cheddar_gpu_ntt_fused_fwd_p1_mont_w{0}_deg{1}", W,
                             degree)
                   .str()
             : llvm::formatv("cheddar_gpu_ntt_fused_fwd_p1_w{0}_deg{1}", W,
                             degree)
                   .str();
  }

  SmallVector<Type> p1ArgTypes = {dynMemrefType, dynMemrefType, dynMemrefType};
  if (mont) p1ArgTypes.push_back(dyn1DMemrefType);
  if (hasSkipCopy) p1ArgTypes.push_back(dynMemrefType);
  p1ArgTypes.push_back(dyn1DIndexMemrefType);
  p1ArgTypes.append({indexType, indexType, indexType, indexType, indexType});
  if (hasSkipCopy) p1ArgTypes.push_back(indexType);
  p1ArgTypes.push_back(i1Type);

  FunctionType p1KernelType = FunctionType::get(context, p1ArgTypes, {});

  gpu::GPUFuncOp p1Kernel = getOrCreateKernelWithSharedMem(
      gpuMod, p1KernelName, p1KernelType, /*smemElements=*/2048,
      /*smemElemType=*/intType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock, Value smem) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        buildNTTFwdP1Body(ib, entryBlock, smem, W, totalLimbs, mont,
                          hasSkipCopy);
      });

  SmallVector<Value> p1LaunchOperands = {outDyn, inDyn, twiddlesDyn};
  if (mont) {
    std::string r2Name = llvm::formatv("__cheddar_r2_w{0}", W).str();
    auto r2Global = module.lookupSymbol<memref::GlobalOp>(r2Name);
    Value r2Val = getOrCreateHostGetGlobal(b, loc, r2Global, op.getOperation());
    Value r2Dyn = castToDynamic1D(b, loc, r2Val);
    p1LaunchOperands.push_back(r2Dyn);
  }
  if (hasSkipCopy) {
    Value skipSrcDyn = castToDynamicFirstDim(b, loc, op.getSkipCopySrc());
    p1LaunchOperands.push_back(skipSrcDyn);
  }
  p1LaunchOperands.push_back(primeIndicesDyn);
  p1LaunchOperands.append(
      {outStartVal, inStartVal, primeOffsetVal, skipStartVal, skipEndVal});
  if (hasSkipCopy) {
    Value skipInOffVal =
        b.create<arith::ConstantIndexOp>(loc, op.getSkipCopyInOffset());
    p1LaunchOperands.push_back(skipInOffVal);
  }
  p1LaunchOperands.push_back(hasPrimeIndicesVal);

  emitKernelLaunch(b, loc, p1Kernel, gridP1, blockP1, p1LaunchOperands);

  // === Phase 2 Launch ===
  gpu::KernelDim3 gridP2{c128, cLCount, c1};
  gpu::KernelDim3 blockP2{c64, c1, c1};

  std::string p2KernelName =
      llvm::formatv("cheddar_gpu_ntt_fused_fwd_p2_w{0}_deg{1}", W, degree)
          .str();
  FunctionType p2KernelType =
      FunctionType::get(context,
                        {dynMemrefType, dynMemrefType, dyn1DIndexMemrefType,
                         indexType, indexType, indexType, indexType, i1Type},
                        {});

  gpu::GPUFuncOp p2Kernel = getOrCreateKernelWithSharedMem(
      gpuMod, p2KernelName, p2KernelType, /*smemElements=*/512,
      /*smemElemType=*/intType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock, Value smem) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value twArg = entryBlock->getArgument(1);
        Value pIndicesArg = entryBlock->getArgument(2);
        Value oStartArg = entryBlock->getArgument(3);
        Value pOffsetArg = entryBlock->getArgument(4);
        Value sStartArg = entryBlock->getArgument(5);
        Value sEndArg = entryBlock->getArgument(6);
        Value hIndicesArg = entryBlock->getArgument(7);

        Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
        Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
        Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);

        Value localLimb = by;
        Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);

        Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge, outLimb,
                                            sStartArg);
        Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt, outLimb,
                                            sEndArg);
        Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
        Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
        Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);

        auto ifOp = ib.create<scf::IfOp>(notSkipped, /*withElseRegion=*/false);
        ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

        Value c0 = ib.create<arith::ConstantIndexOp>(0);
        Value primeOffsetLimb = ib.create<arith::AddIOp>(localLimb, pOffsetArg);
        Value safeIdx = ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
        Value primeFromIdx =
            ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
        Value absLimb = ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx,
                                                   primeOffsetLimb);

        auto [primesMemref, qInvMemref] =
            getDevicePrimeGlobals(ib, W, totalLimbs);
        Value q = ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
        Value qInv = ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

        auto getTwiddle = [&](Value twIdxVal) -> Value {
          return ib.create<memref::LoadOp>(twArg,
                                           ValueRange{absLimb, twIdxVal});
        };

        Value c3 = ib.create<arith::ConstantIndexOp>(3);
        Value c6 = ib.create<arith::ConstantIndexOp>(6);
        Value c7 = ib.create<arith::ConstantIndexOp>(7);
        Value c9 = ib.create<arith::ConstantIndexOp>(9);
        Value bxShL6 = ib.create<arith::ShLIOp>(bx, c6);
        Value xIdx = ib.create<arith::AddIOp>(bxShL6, tx);

        // Load 8 elements from outArg (written by Phase 1)
        Value bxShL9 = ib.create<arith::ShLIOp>(bx, c9);
        Value loadBase = ib.create<arith::AddIOp>(tx, bxShL9);
        Value local[8];
        for (int i = 0; i < 8; ++i) {
          Value offset = ib.create<arith::ConstantIndexOp>(i << 6);
          Value srcIdx = ib.create<arith::AddIOp>(loadBase, offset);
          local[i] =
              ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, srcIdx});
        }

        // Stages 8..10: MultiRadixNTTFirst<word, 8, 3>(local, tw_idx = 128 +
        // bx)
        Value c128Const = ib.create<arith::ConstantIndexOp>(128);
        Value twIdx0 = ib.create<arith::AddIOp>(c128Const, bx);

        // Stage 8
        {
          Value w = getTwiddle(twIdx0);
          for (int j = 0; j < 4; ++j) {
            butterflyNTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
          }
        }
        // Stage 9
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value twIdxShL1 = ib.create<arith::ShLIOp>(twIdx0, c1Const);
          Value w0 = getTwiddle(twIdxShL1);
          Value w1Idx = ib.create<arith::AddIOp>(twIdxShL1, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 2; ++j) {
            butterflyNTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
            butterflyNTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
          }
        }
        // Stage 10
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value twIdxShL2 = ib.create<arith::ShLIOp>(twIdx0, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(twIdxShL2, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                qInv);
          }
        }

        // Store to smem: temp[tx + (j << 6)] = local[j]
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 6);
          Value smIdx = ib.create<arith::AddIOp>(tx, smOffset);
          ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
        }
        ib.create<gpu::BarrierOp>();

        // Transposed reload: sm_idx = ((tx >> 3) << 6) + (tx & 7)
        Value txShR3 = ib.create<arith::ShRUIOp>(tx, c3);
        Value txShR3ShL6 = ib.create<arith::ShLIOp>(txShR3, c6);
        Value txAnd7 = ib.create<arith::AndIOp>(tx, c7);
        Value smIdx1Base = ib.create<arith::AddIOp>(txShR3ShL6, txAnd7);
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 3);
          Value smIdx = ib.create<arith::AddIOp>(smIdx1Base, smOffset);
          local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
        }

        // Stages 11..13: MultiRadixNTT<word, 8, 3>(local, tw_idx = (8192 +
        // x_idx) >> 3)
        Value c8192 = ib.create<arith::ConstantIndexOp>(8192);
        Value xIdxPlus8192 = ib.create<arith::AddIOp>(c8192, xIdx);
        Value twIdx1 = ib.create<arith::ShRUIOp>(xIdxPlus8192, c3);

        // Stage 11
        {
          Value w = getTwiddle(twIdx1);
          for (int j = 0; j < 4; ++j) {
            butterflyNTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
          }
        }
        // Stage 12
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value twIdxShL1 = ib.create<arith::ShLIOp>(twIdx1, c1Const);
          Value w0 = getTwiddle(twIdxShL1);
          Value w1Idx = ib.create<arith::AddIOp>(twIdxShL1, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 2; ++j) {
            butterflyNTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
            butterflyNTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
          }
        }
        // Stage 13
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value twIdxShL2 = ib.create<arith::ShLIOp>(twIdx1, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(twIdxShL2, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                qInv);
          }
        }

        // Store back to smem
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 3);
          Value smIdx = ib.create<arith::AddIOp>(smIdx1Base, smOffset);
          ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
        }
        ib.create<gpu::BarrierOp>();

        // Reload: sm_idx0 = tx << 3
        Value txShL3 = ib.create<arith::ShLIOp>(tx, c3);
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j);
          Value smIdx = ib.create<arith::AddIOp>(txShL3, smOffset);
          local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
        }

        // Stages 14..16: MultiRadixNTT<word, 8, 3>(local, tw_idx = 8192 +
        // x_idx)
        Value twIdx2 = xIdxPlus8192;
        // Stage 14
        {
          Value w = getTwiddle(twIdx2);
          for (int j = 0; j < 4; ++j) {
            butterflyNTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
          }
        }
        // Stage 15
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value twIdxShL1 = ib.create<arith::ShLIOp>(twIdx2, c1Const);
          Value w0 = getTwiddle(twIdxShL1);
          Value w1Idx = ib.create<arith::AddIOp>(twIdxShL1, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 2; ++j) {
            butterflyNTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
            butterflyNTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
          }
        }
        // Stage 16
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value twIdxShL2 = ib.create<arith::ShLIOp>(twIdx2, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(twIdxShL2, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                qInv);
          }
        }

        // Store to outArg
        Value xIdxShL3 = ib.create<arith::ShLIOp>(xIdx, c3);
        for (int j = 0; j < 8; ++j) {
          Value offset = ib.create<arith::ConstantIndexOp>(j);
          Value dstIdx = ib.create<arith::AddIOp>(xIdxShL3, offset);
          ib.create<memref::StoreOp>(local[j], outArg,
                                     ValueRange{outLimb, dstIdx});
        }

        ib.setInsertionPointAfter(ifOp);
        ib.create<gpu::ReturnOp>();
      });

  emitKernelLaunch(
      b, loc, p2Kernel, gridP2, blockP2,
      ValueRange{outDyn, twiddlesDyn, primeIndicesDyn, outStartVal,
                 primeOffsetVal, skipStartVal, skipEndVal, hasPrimeIndicesVal});
}

template <unsigned W>
void rewriteINTTFused(INTTOp op, PatternRewriter& rewriter,
                      ParamsAttr paramsAttr, int64_t inStart, int64_t outStart,
                      int64_t primeStart, int64_t lCount) {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  MLIRContext* context = b.getContext();
  Location loc = op.getLoc();

  Type intType = IntegerType::get(context, W);
  Type indexType = b.getIndexType();
  Type i1Type = b.getI1Type();

  int64_t degree = 1LL << paramsAttr.getLogN();
  size_t totalLimbs = paramsAttr.getNumLimbs();

  MemRefType dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, intType);
  MemRefType dyn1DMemrefType = MemRefType::get({ShapedType::kDynamic}, intType);
  MemRefType dyn1DIndexMemrefType =
      MemRefType::get({ShapedType::kDynamic}, indexType);

  getOrCreateModuleTwiddleGlobals<W>(module, paramsAttr, degree);

  std::string invName =
      llvm::formatv("__cheddar_twiddle_inv_w{0}_deg{1}", W, degree).str();
  auto invGlobal = module.lookupSymbol<memref::GlobalOp>(invName);
  Value twiddlesVal =
      getOrCreateHostGetGlobal(b, loc, invGlobal, op.getOperation());
  Value twiddlesDyn = castToDynamicFirstDim(b, loc, twiddlesVal);

  memref::GlobalOp primeIndicesGlobal;
  bool hasPrimeIndices = false;
  int64_t primeOffset = primeStart;
  if (op.getPrimeIndices()) {
    primeIndicesGlobal =
        getOrCreatePrimeIndicesGlobal(module, *op.getPrimeIndices());
    hasPrimeIndices = true;
    primeOffset = 0;
  } else {
    primeIndicesGlobal =
        module.lookupSymbol<memref::GlobalOp>("__cheddar_dummy_indices");
  }
  Value primeIndicesVal =
      getOrCreateHostGetGlobal(b, loc, primeIndicesGlobal, op.getOperation());
  Value primeIndicesDyn = castToDynamic1D(b, loc, primeIndicesVal);

  int64_t inSplitCnt = lCount;
  int64_t inExtraLimbs = 0;
  if (op.getInSplit() && op.getInSplit()->size() == 2) {
    inSplitCnt = (*op.getInSplit())[0];
    inExtraLimbs = (*op.getInSplit())[1];
  }
  Value inSplitCntVal = b.create<arith::ConstantIndexOp>(loc, inSplitCnt);
  Value inExtraLimbsVal = b.create<arith::ConstantIndexOp>(loc, inExtraLimbs);

  Value outDyn = castToDynamicFirstDim(b, loc, op.getOut());
  Value inDyn = castToDynamicFirstDim(b, loc, op.getIn());
  Value outStartVal = b.create<arith::ConstantIndexOp>(loc, outStart);
  Value inStartVal = b.create<arith::ConstantIndexOp>(loc, inStart);
  Value primeOffsetVal = b.create<arith::ConstantIndexOp>(loc, primeOffset);
  Value skipStartVal = b.create<arith::ConstantIndexOp>(loc, 0);
  Value skipEndVal = b.create<arith::ConstantIndexOp>(loc, 0);
  Value hasPrimeIndicesVal =
      b.create<arith::ConstantIntOp>(loc, hasPrimeIndices ? 1 : 0, 1);

  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value cLCount = b.create<arith::ConstantIndexOp>(loc, lCount);
  Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
  Value c64 = b.create<arith::ConstantIndexOp>(loc, 64);
  Value c128 = b.create<arith::ConstantIndexOp>(loc, 128);

  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  // === Phase 1 Launch: Grid (128, lCount, 1) ===
  gpu::KernelDim3 gridP1{c128, cLCount, c1};
  gpu::KernelDim3 blockP1{c64, c1, c1};

  std::string p1KernelName =
      llvm::formatv("cheddar_gpu_intt_fused_p1_w{0}_deg{1}", W, degree).str();
  FunctionType p1KernelType =
      FunctionType::get(context,
                        {dynMemrefType, dynMemrefType, dynMemrefType,
                         dyn1DIndexMemrefType, indexType, indexType, indexType,
                         indexType, indexType, indexType, indexType, i1Type},
                        {});

  gpu::GPUFuncOp p1Kernel = getOrCreateKernelWithSharedMem(
      gpuMod, p1KernelName, p1KernelType, /*smemElements=*/512,
      /*smemElemType=*/intType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock, Value smem) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value inArg = entryBlock->getArgument(1);
        Value twArg = entryBlock->getArgument(2);
        Value pIndicesArg = entryBlock->getArgument(3);
        Value oStartArg = entryBlock->getArgument(4);
        Value iStartArg = entryBlock->getArgument(5);
        Value pOffsetArg = entryBlock->getArgument(6);
        Value sStartArg = entryBlock->getArgument(7);
        Value sEndArg = entryBlock->getArgument(8);
        Value inSplitCntArg = entryBlock->getArgument(9);
        Value inExtraLimbsArg = entryBlock->getArgument(10);
        Value hIndicesArg = entryBlock->getArgument(11);

        Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
        Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
        Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);

        Value localLimb = by;
        Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);
        Value c0 = ib.create<arith::ConstantIndexOp>(0);
        Value geSplit = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                                 localLimb, inSplitCntArg);
        Value extraShift =
            ib.create<arith::SelectOp>(geSplit, inExtraLimbsArg, c0);
        Value inLimb = ib.create<arith::AddIOp>(localLimb, iStartArg);
        inLimb = ib.create<arith::AddIOp>(inLimb, extraShift);

        Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge, outLimb,
                                            sStartArg);
        Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt, outLimb,
                                            sEndArg);
        Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
        Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
        Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);

        auto ifOp = ib.create<scf::IfOp>(notSkipped, /*withElseRegion=*/false);
        ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

        Value primeOffsetLimb = ib.create<arith::AddIOp>(localLimb, pOffsetArg);
        Value safeIdx = ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
        Value primeFromIdx =
            ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
        Value absLimb = ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx,
                                                   primeOffsetLimb);

        auto [primesMemref, qInvMemref] =
            getDevicePrimeGlobals(ib, W, totalLimbs);
        Value q = ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
        Value qInv = ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

        auto getTwiddle = [&](Value twIdxVal) -> Value {
          return ib.create<memref::LoadOp>(twArg,
                                           ValueRange{absLimb, twIdxVal});
        };

        Value c3 = ib.create<arith::ConstantIndexOp>(3);
        Value c6 = ib.create<arith::ConstantIndexOp>(6);
        Value c7 = ib.create<arith::ConstantIndexOp>(7);
        Value c9 = ib.create<arith::ConstantIndexOp>(9);
        Value bxShL6 = ib.create<arith::ShLIOp>(bx, c6);
        Value xIdx = ib.create<arith::AddIOp>(bxShL6, tx);

        // Load 8 elements from inArg: src_idx = (x_idx << 3) + j
        Value xIdxShL3 = ib.create<arith::ShLIOp>(xIdx, c3);
        Value local[8];
        for (int j = 0; j < 8; ++j) {
          Value offset = ib.create<arith::ConstantIndexOp>(j);
          Value srcIdx = ib.create<arith::AddIOp>(xIdxShL3, offset);
          local[j] =
              ib.create<memref::LoadOp>(inArg, ValueRange{inLimb, srcIdx});
        }

        // Stages 1..3: MultiRadixINTT<word, 8, 3>(local, tw_idx = 8192 + x_idx)
        Value c8192 = ib.create<arith::ConstantIndexOp>(8192);
        Value twIdx0 = ib.create<arith::AddIOp>(c8192, xIdx);
        // s = 1: stride 1, num_tw 4, curr_tw = tw_idx << 2
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value currTw = ib.create<arith::ShLIOp>(twIdx0, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(currTw, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyINTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                 qInv);
          }
        }
        // s = 2: stride 2, num_tw 2, curr_tw = tw_idx << 1
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value currTw = ib.create<arith::ShLIOp>(twIdx0, c1Const);
          Value w0 = getTwiddle(currTw);
          Value w1Idx = ib.create<arith::AddIOp>(currTw, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 2; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
            butterflyINTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
          }
        }
        // s = 3: stride 4, num_tw 1, curr_tw = tw_idx
        {
          Value w = getTwiddle(twIdx0);
          for (int j = 0; j < 4; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
          }
        }

        // Store to smem: temp[sm_idx + j] where sm_idx = tx << 3
        Value txShL3 = ib.create<arith::ShLIOp>(tx, c3);
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j);
          Value smIdx = ib.create<arith::AddIOp>(txShL3, smOffset);
          ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
        }
        ib.create<gpu::BarrierOp>();

        // Transposed reload: sm_idx = (tx & 7) + ((tx >> 3) << 6)
        Value txShR3 = ib.create<arith::ShRUIOp>(tx, c3);
        Value txShR3ShL6 = ib.create<arith::ShLIOp>(txShR3, c6);
        Value txAnd7 = ib.create<arith::AndIOp>(tx, c7);
        Value smIdx1Base = ib.create<arith::AddIOp>(txShR3ShL6, txAnd7);
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 3);
          Value smIdx = ib.create<arith::AddIOp>(smIdx1Base, smOffset);
          local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
        }

        // Stages 4..6: MultiRadixINTT<word, 8, 3>(local, tw_idx = (8192 +
        // x_idx) >> 3)
        Value twIdx1 = ib.create<arith::ShRUIOp>(twIdx0, c3);
        // s = 1
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value currTw = ib.create<arith::ShLIOp>(twIdx1, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(currTw, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyINTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                 qInv);
          }
        }
        // s = 2
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value currTw = ib.create<arith::ShLIOp>(twIdx1, c1Const);
          Value w0 = getTwiddle(currTw);
          Value w1Idx = ib.create<arith::AddIOp>(currTw, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 2; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
            butterflyINTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
          }
        }
        // s = 3
        {
          Value w = getTwiddle(twIdx1);
          for (int j = 0; j < 4; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
          }
        }

        // Store to smem
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 3);
          Value smIdx = ib.create<arith::AddIOp>(smIdx1Base, smOffset);
          ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
        }
        ib.create<gpu::BarrierOp>();

        // Reload
        for (int j = 0; j < 8; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 6);
          Value smIdx = ib.create<arith::AddIOp>(tx, smOffset);
          local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
        }

        // Stages 7..9: MultiRadixINTTLast<word, 8, 3>(local, tw_idx = 128 + bx)
        Value c128Const = ib.create<arith::ConstantIndexOp>(128);
        Value twIdx2 = ib.create<arith::AddIOp>(c128Const, bx);
        // s = 3
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value currTw = ib.create<arith::ShLIOp>(twIdx2, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(currTw, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyINTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                 qInv);
          }
        }
        // s = 2
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value currTw = ib.create<arith::ShLIOp>(twIdx2, c1Const);
          Value w0 = getTwiddle(currTw);
          Value w1Idx = ib.create<arith::AddIOp>(currTw, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 2; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
            butterflyINTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
          }
        }
        // s = 1
        {
          Value w = getTwiddle(twIdx2);
          for (int j = 0; j < 4; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
          }
        }

        // Store to outArg: dst_idx = tx + (bx << 9) + (j << 6)
        Value bxShL9 = ib.create<arith::ShLIOp>(bx, c9);
        Value dstBase = ib.create<arith::AddIOp>(tx, bxShL9);
        for (int j = 0; j < 8; ++j) {
          Value offset = ib.create<arith::ConstantIndexOp>(j << 6);
          Value dstIdx = ib.create<arith::AddIOp>(dstBase, offset);
          ib.create<memref::StoreOp>(local[j], outArg,
                                     ValueRange{outLimb, dstIdx});
        }

        ib.setInsertionPointAfter(ifOp);
        ib.create<gpu::ReturnOp>();
      });

  emitKernelLaunch(
      b, loc, p1Kernel, gridP1, blockP1,
      ValueRange{outDyn, inDyn, twiddlesDyn, primeIndicesDyn, outStartVal,
                 inStartVal, primeOffsetVal, skipStartVal, skipEndVal,
                 inSplitCntVal, inExtraLimbsVal, hasPrimeIndicesVal});

  // === Phase 2 Launch: Grid (32, lCount, 1), Block (128, 1, 1), SharedMem 2048
  // ===
  gpu::KernelDim3 gridP2{c32, cLCount, c1};
  gpu::KernelDim3 blockP2{c128, c1, c1};

  auto epilogue = op.getEpilogue();
  std::string epilogueVariant;
  Value scalesDyn;

  if (epilogue == INTTEpilogue::None) {
    epilogueVariant = "none";
    std::string nInvName =
        op.getMontgomeryConversion()
            ? llvm::formatv("__cheddar_n_inv_w{0}_deg{1}", W, degree).str()
            : llvm::formatv("__cheddar_n_inv_mont_w{0}_deg{1}", W, degree)
                  .str();
    auto nInvGlobal = module.lookupSymbol<memref::GlobalOp>(nInvName);
    Value nInvVal =
        getOrCreateHostGetGlobal(b, loc, nInvGlobal, op.getOperation());
    scalesDyn = castToDynamic1D(b, loc, nInvVal);
  } else if (epilogue == INTTEpilogue::MultConst) {
    epilogueVariant = "mult_const";
    scalesDyn = castToDynamic1D(b, loc, op.getCst());
  } else {
    epilogueVariant = "mult_const_normalize";
    scalesDyn = castToDynamic1D(b, loc, op.getCst());
  }

  std::string p2KernelName =
      llvm::formatv("cheddar_gpu_intt_fused_p2_{0}_w{1}_deg{2}",
                    epilogueVariant, W, degree)
          .str();
  FunctionType p2KernelType = FunctionType::get(
      context,
      {dynMemrefType, dynMemrefType, dyn1DMemrefType, dyn1DIndexMemrefType,
       indexType, indexType, indexType, indexType, i1Type},
      {});

  gpu::GPUFuncOp p2Kernel = getOrCreateKernelWithSharedMem(
      gpuMod, p2KernelName, p2KernelType, /*smemElements=*/2048,
      /*smemElemType=*/intType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock, Value smem) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value twArg = entryBlock->getArgument(1);
        Value scalesArg = entryBlock->getArgument(2);
        Value pIndicesArg = entryBlock->getArgument(3);
        Value oStartArg = entryBlock->getArgument(4);
        Value pOffsetArg = entryBlock->getArgument(5);
        Value sStartArg = entryBlock->getArgument(6);
        Value sEndArg = entryBlock->getArgument(7);
        Value hIndicesArg = entryBlock->getArgument(8);

        Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
        Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
        Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);

        Value localLimb = by;
        Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);

        Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge, outLimb,
                                            sStartArg);
        Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt, outLimb,
                                            sEndArg);
        Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
        Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
        Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);

        auto ifOp = ib.create<scf::IfOp>(notSkipped, /*withElseRegion=*/false);
        ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

        Value c0 = ib.create<arith::ConstantIndexOp>(0);
        Value primeOffsetLimb = ib.create<arith::AddIOp>(localLimb, pOffsetArg);
        Value safeIdx = ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
        Value primeFromIdx =
            ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
        Value absLimb = ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx,
                                                   primeOffsetLimb);

        auto [primesMemref, qInvMemref] =
            getDevicePrimeGlobals(ib, W, totalLimbs);
        Value q = ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
        Value qInv = ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

        auto getTwiddle = [&](Value twIdxVal) -> Value {
          return ib.create<memref::LoadOp>(twArg,
                                           ValueRange{absLimb, twIdxVal});
        };

        Value c4 = ib.create<arith::ConstantIndexOp>(4);
        Value c8 = ib.create<arith::ConstantIndexOp>(8);
        Value c9 = ib.create<arith::ConstantIndexOp>(9);
        Value c13 = ib.create<arith::ConstantIndexOp>(13);
        Value c15 = ib.create<arith::ConstantIndexOp>(15);

        Value stageGroupIdx = ib.create<arith::ShRUIOp>(tx, c4);
        Value batchIdx = ib.create<arith::AndIOp>(tx, c15);

        // Load 16 elements from outArg: src_idx = batch_idx + (stage_group_idx
        // << 13) + (bx << 4) + (i << 9)
        Value bxShL4 = ib.create<arith::ShLIOp>(bx, c4);
        Value sgShL13 = ib.create<arith::ShLIOp>(stageGroupIdx, c13);
        Value loadBase = ib.create<arith::AddIOp>(batchIdx, sgShL13);
        loadBase = ib.create<arith::AddIOp>(loadBase, bxShL4);

        Value local[16];
        for (int i = 0; i < 16; ++i) {
          Value offset = ib.create<arith::ConstantIndexOp>(i << 9);
          Value srcIdx = ib.create<arith::AddIOp>(loadBase, offset);
          local[i] =
              ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, srcIdx});
        }

        // MultiRadixINTT<word, 16, 4>(local, base_tw = 8 + stage_group_idx)
        Value c8Const = ib.create<arith::ConstantIndexOp>(8);
        Value baseTw = ib.create<arith::AddIOp>(c8Const, stageGroupIdx);

        // Stage 10: s = 1: stride 1, num_tw 8
        {
          Value c3Const = ib.create<arith::ConstantIndexOp>(3);
          Value baseTwShL3 = ib.create<arith::ShLIOp>(baseTw, c3Const);
          for (int k = 0; k < 8; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(baseTwShL3, kConst);
            Value wk = getTwiddle(wkIdx);
            butterflyINTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q,
                                 qInv);
          }
        }
        // Stage 11: s = 2: stride 2, num_tw 4
        {
          Value c2Const = ib.create<arith::ConstantIndexOp>(2);
          Value baseTwShL2 = ib.create<arith::ShLIOp>(baseTw, c2Const);
          for (int k = 0; k < 4; ++k) {
            Value kConst = ib.create<arith::ConstantIndexOp>(k);
            Value wkIdx = ib.create<arith::AddIOp>(baseTwShL2, kConst);
            Value wk = getTwiddle(wkIdx);
            for (int j = 0; j < 2; ++j) {
              butterflyINTTRuntime(ib, local[4 * k + j], local[4 * k + 2 + j],
                                   wk, q, qInv);
            }
          }
        }
        // Stage 12: s = 3: stride 4, num_tw 2
        {
          Value c1Const = ib.create<arith::ConstantIndexOp>(1);
          Value currTw = ib.create<arith::ShLIOp>(baseTw, c1Const);
          Value w0 = getTwiddle(currTw);
          Value w1Idx = ib.create<arith::AddIOp>(currTw, c1Const);
          Value w1 = getTwiddle(w1Idx);
          for (int j = 0; j < 4; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 4], w0, q, qInv);
            butterflyINTTRuntime(ib, local[8 + j], local[12 + j], w1, q, qInv);
          }
        }
        // Stage 13: s = 4: stride 8, num_tw 1
        {
          Value w = getTwiddle(baseTw);
          for (int j = 0; j < 8; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 8], w, q, qInv);
          }
        }

        // Store to smem: temp[sm_idx + (j << 4)] = local[j] where sm_idx =
        // batch_idx + (stage_group_idx << 8)
        Value sgShL8 = ib.create<arith::ShLIOp>(stageGroupIdx, c8);
        Value smStoreBase = ib.create<arith::AddIOp>(sgShL8, batchIdx);
        for (int j = 0; j < 16; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 4);
          Value smIdx = ib.create<arith::AddIOp>(smStoreBase, smOffset);
          ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
        }
        ib.create<gpu::BarrierOp>();

        // Reload: local[j] = temp[tx + (j << 7)]
        for (int j = 0; j < 16; ++j) {
          Value smOffset = ib.create<arith::ConstantIndexOp>(j << 7);
          Value smIdx = ib.create<arith::AddIOp>(tx, smOffset);
          local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
        }

        // MultiRadixINTTLast<word, 16, 3>(local, tw_idx = 1)
        // Stage 14: s = 3
        {
          for (int k = 0; k < 4; ++k) {
            Value wk = getTwiddle(ib.create<arith::ConstantIndexOp>(4 + k));
            for (int j = 0; j < 2; ++j) {
              butterflyINTTRuntime(ib, local[4 * k + j], local[4 * k + 2 + j],
                                   wk, q, qInv);
            }
          }
        }
        // Stage 15: s = 2
        {
          Value w0 = getTwiddle(ib.create<arith::ConstantIndexOp>(2));
          Value w1 = getTwiddle(ib.create<arith::ConstantIndexOp>(3));
          for (int j = 0; j < 4; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 4], w0, q, qInv);
            butterflyINTTRuntime(ib, local[8 + j], local[12 + j], w1, q, qInv);
          }
        }
        // Stage 16: s = 1
        {
          Value w = getTwiddle(ib.create<arith::ConstantIndexOp>(1));
          for (int j = 0; j < 8; ++j) {
            butterflyINTTRuntime(ib, local[j], local[j + 8], w, q, qInv);
          }
        }

        // Epilogue scaling on local[j]
        Value scaleVal;
        if (epilogueVariant == "none") {
          scaleVal = ib.create<memref::LoadOp>(scalesArg, ValueRange{absLimb});
        } else {
          scaleVal =
              ib.create<memref::LoadOp>(scalesArg, ValueRange{localLimb});
        }

        for (int j = 0; j < 16; ++j) {
          Value scaled = buildMontMulRuntime(ib, local[j], scaleVal, q, qInv);
          if (epilogueVariant == "mult_const_normalize") {
            scaled = buildCenteredRuntime(ib, scaled, q);
          }
          local[j] = scaled;
        }

        // Store to outArg: dst_idx = batch_idx + (bx << 4) + (stage_group_idx
        // << 9) + (j << 12)
        Value sgShL9 = ib.create<arith::ShLIOp>(stageGroupIdx, c9);
        Value dstBase = ib.create<arith::AddIOp>(batchIdx, bxShL4);
        dstBase = ib.create<arith::AddIOp>(dstBase, sgShL9);
        for (int j = 0; j < 16; ++j) {
          Value offset = ib.create<arith::ConstantIndexOp>(j << 12);
          Value dstIdx = ib.create<arith::AddIOp>(dstBase, offset);
          ib.create<memref::StoreOp>(local[j], outArg,
                                     ValueRange{outLimb, dstIdx});
        }

        ib.setInsertionPointAfter(ifOp);
        ib.create<gpu::ReturnOp>();
      });

  emitKernelLaunch(
      b, loc, p2Kernel, gridP2, blockP2,
      ValueRange{outDyn, twiddlesDyn, scalesDyn, primeIndicesDyn, outStartVal,
                 primeOffsetVal, skipStartVal, skipEndVal, hasPrimeIndicesVal});
}

struct ConvertNTTOp : public OpRewritePattern<NTTOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent ModuleOp");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    auto inType = cast<MemRefType>(op.getIn().getType());
    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));
    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));
    int64_t lCount = op.getOutLimbRange() ? outCount : inCount;
    auto [primeStart, primeCount] = getLimbRange(op.getPrimeRange(), lCount);
    if (!op.getPrimeRange()) {
      primeStart = op.getInLimbRange() ? inStart : outStart;
    }

    if (params.getWord() == 32) {
      rewriteNTTFused<32>(op, rewriter, params, inStart, outStart, primeStart,
                          lCount);
    } else if (params.getWord() == 64) {
      rewriteNTTFused<64>(op, rewriter, params, inStart, outStart, primeStart,
                          lCount);
    } else {
      return op.emitOpError("unsupported word size ") << params.getWord();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertINTTOp : public OpRewritePattern<INTTOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(INTTOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent ModuleOp");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    auto inType = cast<MemRefType>(op.getIn().getType());
    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));
    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));
    int64_t lCount = op.getOutLimbRange() ? outCount : inCount;
    auto [primeStart, primeCount] = getLimbRange(op.getPrimeRange(), lCount);
    if (!op.getPrimeRange()) {
      primeStart = op.getInLimbRange() ? inStart : outStart;
    }

    if (params.getWord() == 32) {
      rewriteINTTFused<32>(op, rewriter, params, inStart, outStart, primeStart,
                           lCount);
    } else if (params.getWord() == 64) {
      rewriteINTTFused<64>(op, rewriter, params, inStart, outStart, primeStart,
                           lCount);
    } else {
      return op.emitOpError("unsupported word size ") << params.getWord();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

enum class ModDownVariant {
  NoSrc2,
  Src2,
  Src2Pad,
};

void buildNTTModDownFwdP2Body(ImplicitLocOpBuilder& ib, Block* entryBlock,
                              Value smem, unsigned W, size_t totalLimbs,
                              ModDownVariant variant) {
  Type intType = IntegerType::get(ib.getContext(), W);
  Value outArg = entryBlock->getArgument(0);
  Value tmpArg = entryBlock->getArgument(1);
  Value twArg = entryBlock->getArgument(2);
  Value invPProdArg = entryBlock->getArgument(3);
  int argIdx = 4;
  Value src2Arg = (variant != ModDownVariant::NoSrc2)
                      ? entryBlock->getArgument(argIdx++)
                      : Value();
  Value padArg = (variant == ModDownVariant::Src2Pad)
                     ? entryBlock->getArgument(argIdx++)
                     : Value();
  Value pIndicesArg = entryBlock->getArgument(argIdx++);
  Value padStartArg = entryBlock->getArgument(argIdx++);
  Value padEndArg = entryBlock->getArgument(argIdx++);
  Value s2OffArg = (variant != ModDownVariant::NoSrc2)
                       ? entryBlock->getArgument(argIdx++)
                       : Value();
  Value pOffsetArg = entryBlock->getArgument(argIdx++);
  Value hIndicesArg = entryBlock->getArgument(argIdx++);

  Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
  Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
  Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);

  Value localLimb = by;
  Value c0 = ib.create<arith::ConstantIndexOp>(0);
  Value primeOffsetLimb = ib.create<arith::AddIOp>(localLimb, pOffsetArg);
  Value safeIdx = ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
  Value primeFromIdx =
      ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
  Value absLimb =
      ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx, primeOffsetLimb);

  auto [primesMemref, qInvMemref] = getDevicePrimeGlobals(ib, W, totalLimbs);
  Value q = ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
  Value qInv = ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

  auto getTwiddle = [&](Value twIdxVal) -> Value {
    return ib.create<memref::LoadOp>(twArg, ValueRange{absLimb, twIdxVal});
  };

  Value c3 = ib.create<arith::ConstantIndexOp>(3);
  Value c6 = ib.create<arith::ConstantIndexOp>(6);
  Value c7 = ib.create<arith::ConstantIndexOp>(7);
  Value c9 = ib.create<arith::ConstantIndexOp>(9);
  Value bxShL6 = ib.create<arith::ShLIOp>(bx, c6);
  Value xIdx = ib.create<arith::AddIOp>(bxShL6, tx);

  // Load 8 elements from tmpArg (written by Phase 1 into nttTemp)
  Value bxShL9 = ib.create<arith::ShLIOp>(bx, c9);
  Value loadBase = ib.create<arith::AddIOp>(tx, bxShL9);
  Value local[8];
  for (int i = 0; i < 8; ++i) {
    Value offset = ib.create<arith::ConstantIndexOp>(i << 6);
    Value srcIdx = ib.create<arith::AddIOp>(loadBase, offset);
    local[i] = ib.create<memref::LoadOp>(tmpArg, ValueRange{localLimb, srcIdx});
  }

  // Stages 8..10: MultiRadixNTTFirst<word, 8, 3>(local, tw_idx = 128 + bx)
  Value c128Const = ib.create<arith::ConstantIndexOp>(128);
  Value twIdx0 = ib.create<arith::AddIOp>(c128Const, bx);

  // Stage 8
  {
    Value w = getTwiddle(twIdx0);
    for (int j = 0; j < 4; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
    }
  }
  // Stage 9
  {
    Value c1Const = ib.create<arith::ConstantIndexOp>(1);
    Value twIdxShL1 = ib.create<arith::ShLIOp>(twIdx0, c1Const);
    Value w0 = getTwiddle(twIdxShL1);
    Value w1Idx = ib.create<arith::AddIOp>(twIdxShL1, c1Const);
    Value w1 = getTwiddle(w1Idx);
    for (int j = 0; j < 2; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
      butterflyNTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
    }
  }
  // Stage 10
  {
    Value c2Const = ib.create<arith::ConstantIndexOp>(2);
    Value twIdxShL2 = ib.create<arith::ShLIOp>(twIdx0, c2Const);
    for (int k = 0; k < 4; ++k) {
      Value kConst = ib.create<arith::ConstantIndexOp>(k);
      Value wkIdx = ib.create<arith::AddIOp>(twIdxShL2, kConst);
      Value wk = getTwiddle(wkIdx);
      butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q, qInv);
    }
  }

  // Store to smem: temp[tx + (j << 6)] = local[j]
  for (int j = 0; j < 8; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j << 6);
    Value smIdx = ib.create<arith::AddIOp>(tx, smOffset);
    ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
  }
  ib.create<gpu::BarrierOp>();

  // Transposed reload: sm_idx = ((tx >> 3) << 6) + (tx & 7)
  Value txShR3 = ib.create<arith::ShRUIOp>(tx, c3);
  Value txShR3ShL6 = ib.create<arith::ShLIOp>(txShR3, c6);
  Value txAnd7 = ib.create<arith::AndIOp>(tx, c7);
  Value smIdx1Base = ib.create<arith::AddIOp>(txShR3ShL6, txAnd7);
  for (int j = 0; j < 8; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j << 3);
    Value smIdx = ib.create<arith::AddIOp>(smIdx1Base, smOffset);
    local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
  }

  // Stages 11..13: MultiRadixNTT<word, 8, 3>(local, tw_idx = (8192 + x_idx) >>
  // 3)
  Value c8192 = ib.create<arith::ConstantIndexOp>(8192);
  Value xIdxPlus8192 = ib.create<arith::AddIOp>(c8192, xIdx);
  Value twIdx1 = ib.create<arith::ShRUIOp>(xIdxPlus8192, c3);

  // Stage 11
  {
    Value w = getTwiddle(twIdx1);
    for (int j = 0; j < 4; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
    }
  }
  // Stage 12
  {
    Value c1Const = ib.create<arith::ConstantIndexOp>(1);
    Value twIdxShL1 = ib.create<arith::ShLIOp>(twIdx1, c1Const);
    Value w0 = getTwiddle(twIdxShL1);
    Value w1Idx = ib.create<arith::AddIOp>(twIdxShL1, c1Const);
    Value w1 = getTwiddle(w1Idx);
    for (int j = 0; j < 2; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
      butterflyNTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
    }
  }
  // Stage 13
  {
    Value c2Const = ib.create<arith::ConstantIndexOp>(2);
    Value twIdxShL2 = ib.create<arith::ShLIOp>(twIdx1, c2Const);
    for (int k = 0; k < 4; ++k) {
      Value kConst = ib.create<arith::ConstantIndexOp>(k);
      Value wkIdx = ib.create<arith::AddIOp>(twIdxShL2, kConst);
      Value wk = getTwiddle(wkIdx);
      butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q, qInv);
    }
  }

  // Store back to smem
  for (int j = 0; j < 8; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j << 3);
    Value smIdx = ib.create<arith::AddIOp>(smIdx1Base, smOffset);
    ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
  }
  ib.create<gpu::BarrierOp>();

  // Reload: sm_idx0 = tx << 3
  Value txShL3 = ib.create<arith::ShLIOp>(tx, c3);
  for (int j = 0; j < 8; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j);
    Value smIdx = ib.create<arith::AddIOp>(txShL3, smOffset);
    local[j] = ib.create<memref::LoadOp>(smem, ValueRange{smIdx});
  }

  // Stages 14..16: MultiRadixNTT<word, 8, 3>(local, tw_idx = 8192 + x_idx)
  Value twIdx2 = xIdxPlus8192;
  // Stage 14
  {
    Value w = getTwiddle(twIdx2);
    for (int j = 0; j < 4; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 4], w, q, qInv);
    }
  }
  // Stage 15
  {
    Value c1Const = ib.create<arith::ConstantIndexOp>(1);
    Value twIdxShL1 = ib.create<arith::ShLIOp>(twIdx2, c1Const);
    Value w0 = getTwiddle(twIdxShL1);
    Value w1Idx = ib.create<arith::AddIOp>(twIdxShL1, c1Const);
    Value w1 = getTwiddle(w1Idx);
    for (int j = 0; j < 2; ++j) {
      butterflyNTTRuntime(ib, local[j], local[j + 2], w0, q, qInv);
      butterflyNTTRuntime(ib, local[4 + j], local[6 + j], w1, q, qInv);
    }
  }
  // Stage 16
  {
    Value c2Const = ib.create<arith::ConstantIndexOp>(2);
    Value twIdxShL2 = ib.create<arith::ShLIOp>(twIdx2, c2Const);
    for (int k = 0; k < 4; ++k) {
      Value kConst = ib.create<arith::ConstantIndexOp>(k);
      Value wkIdx = ib.create<arith::AddIOp>(twIdxShL2, kConst);
      Value wk = getTwiddle(wkIdx);
      butterflyNTTRuntime(ib, local[2 * k], local[2 * k + 1], wk, q, qInv);
    }
  }

  // Write local[0..7] back to smem[(tx << 3) + j]
  for (int j = 0; j < 8; ++j) {
    Value smOffset = ib.create<arith::ConstantIndexOp>(j);
    Value smIdx = ib.create<arith::AddIOp>(txShL3, smOffset);
    ib.create<memref::StoreOp>(local[j], smem, ValueRange{smIdx});
  }
  ib.create<gpu::BarrierOp>();

  Value zeroVal = ib.create<arith::ConstantIntOp>(intType, 0);
  Value invPProd = ib.create<memref::LoadOp>(invPProdArg, ValueRange{by});
  Value isGe =
      ib.create<arith::CmpIOp>(arith::CmpIPredicate::uge, by, padStartArg);
  Value isLt =
      ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult, by, padEndArg);
  Value inPad = ib.create<arith::AndIOp>(isGe, isLt);

  Value padIdx = ib.create<arith::SubIOp>(by, padStartArg);
  Value safePadIdx = ib.create<arith::SelectOp>(inPad, padIdx, c0);
  Value s2Limb = (variant != ModDownVariant::NoSrc2)
                     ? ib.create<arith::AddIOp>(safePadIdx, s2OffArg)
                     : Value();
  Value padVal = zeroVal;
  if (variant == ModDownVariant::Src2Pad) {
    Value padValLoaded =
        ib.create<memref::LoadOp>(padArg, ValueRange{safePadIdx});
    padVal = ib.create<arith::SelectOp>(inPad, padValLoaded, zeroVal);
  }

  for (int k = 0; k < 8; ++k) {
    Value k64 = ib.create<arith::ConstantIndexOp>(k * 64);
    Value i = ib.create<arith::AddIOp>(tx, k64);
    Value coeffIdx = ib.create<arith::AddIOp>(bxShL9, i);
    Value nttVal = ib.create<memref::LoadOp>(smem, ValueRange{i});

    Value term1;
    if (variant == ModDownVariant::NoSrc2) {
      term1 = zeroVal;
    } else if (variant == ModDownVariant::Src2) {
      Value s2Val =
          ib.create<memref::LoadOp>(src2Arg, ValueRange{s2Limb, coeffIdx});
      term1 = ib.create<arith::SelectOp>(inPad, s2Val, zeroVal);
    } else {
      Value s2Val =
          ib.create<memref::LoadOp>(src2Arg, ValueRange{s2Limb, coeffIdx});
      Value s2Mont = buildMontMulRuntime(ib, s2Val, padVal, q, qInv);
      term1 = ib.create<arith::SelectOp>(inPad, s2Mont, zeroVal);
    }

    Value diff = buildModSubRuntime(ib, term1, nttVal, q);
    Value res = buildMontMulRuntime(ib, diff, invPProd, q, qInv);
    ib.create<memref::StoreOp>(res, outArg, ValueRange{by, coeffIdx});
  }

  ib.create<gpu::ReturnOp>();
}

template <unsigned W>
void rewriteNTTModDownFused(NTTModDownOp op, PatternRewriter& rewriter,
                            ParamsAttr paramsAttr) {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  MLIRContext* context = b.getContext();
  Location loc = op.getLoc();

  Type intType = IntegerType::get(context, W);
  Type indexType = b.getIndexType();
  Type i1Type = b.getI1Type();

  int64_t degree = 1LL << paramsAttr.getLogN();
  size_t totalLimbs = paramsAttr.getNumLimbs();

  auto outType = cast<MemRefType>(op.getOut().getType());
  int64_t dstLen = outType.getDimSize(0);

  MemRefType dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, intType);
  MemRefType dyn1DMemrefType = MemRefType::get({ShapedType::kDynamic}, intType);
  MemRefType dyn1DIndexMemrefType =
      MemRefType::get({ShapedType::kDynamic}, indexType);

  // Scratch buffer for Phase 1 output -> Phase 2 input
  auto tempType = MemRefType::get({dstLen, degree}, intType);
  Value nttTemp = memref::AllocOp::create(b, tempType);
  Value nttTempDyn = castToDynamicFirstDim(b, loc, nttTemp);

  getOrCreateModuleTwiddleGlobals<W>(module, paramsAttr, degree);

  std::string fwdName =
      llvm::formatv("__cheddar_twiddle_fwd_w{0}_deg{1}", W, degree).str();
  auto fwdGlobal = module.lookupSymbol<memref::GlobalOp>(fwdName);
  Value twiddlesVal =
      getOrCreateHostGetGlobal(b, loc, fwdGlobal, op.getOperation());
  Value twiddlesDyn = castToDynamicFirstDim(b, loc, twiddlesVal);

  memref::GlobalOp primeIndicesGlobal;
  bool hasPrimeIndices = false;
  int64_t primeOffset = op.getPrimeOffset();
  if (op.getPrimeIndices()) {
    primeIndicesGlobal =
        getOrCreatePrimeIndicesGlobal(module, *op.getPrimeIndices());
    hasPrimeIndices = true;
    primeOffset = 0;
  } else {
    primeIndicesGlobal =
        module.lookupSymbol<memref::GlobalOp>("__cheddar_dummy_indices");
  }
  Value primeIndicesVal =
      getOrCreateHostGetGlobal(b, loc, primeIndicesGlobal, op.getOperation());
  Value primeIndicesDyn = castToDynamic1D(b, loc, primeIndicesVal);

  Value outDyn = castToDynamicFirstDim(b, loc, op.getOut());
  Value nttInDyn = castToDynamicFirstDim(b, loc, op.getNttIn());
  Value invPProdDyn = castToDynamic1D(b, loc, op.getInvPrimeProd());

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
  Value c64 = b.create<arith::ConstantIndexOp>(loc, 64);
  Value c128 = b.create<arith::ConstantIndexOp>(loc, 128);
  Value cDstLen = b.create<arith::ConstantIndexOp>(loc, dstLen);

  Value padStartVal = b.create<arith::ConstantIndexOp>(loc, op.getPadStart());
  Value padEndVal = b.create<arith::ConstantIndexOp>(loc, op.getPadEnd());
  Value s2OffVal = b.create<arith::ConstantIndexOp>(loc, op.getSrc2Offset());
  Value primeOffsetVal = b.create<arith::ConstantIndexOp>(loc, primeOffset);
  Value hasPrimeIndicesVal =
      b.create<arith::ConstantIntOp>(loc, hasPrimeIndices ? 1 : 0, 1);

  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  // === Phase 1 Launch: ntt_in -> nttTemp ===
  gpu::KernelDim3 gridP1{c32, cDstLen, c1};
  gpu::KernelDim3 blockP1{c128, c1, c1};

  std::string p1KernelName =
      llvm::formatv("cheddar_gpu_ntt_fused_fwd_p1_w{0}_deg{1}", W, degree)
          .str();
  SmallVector<Type> p1ArgTypes = {
      dynMemrefType, dynMemrefType, dynMemrefType, dyn1DIndexMemrefType,
      indexType,     indexType,     indexType,     indexType,
      indexType,     i1Type};
  FunctionType p1KernelType = FunctionType::get(context, p1ArgTypes, {});
  gpu::GPUFuncOp p1Kernel = getOrCreateKernelWithSharedMem(
      gpuMod, p1KernelName, p1KernelType, /*smemElements=*/2048,
      /*smemElemType=*/intType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock, Value smem) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        buildNTTFwdP1Body(ib, entryBlock, smem, W, totalLimbs,
                          /*mont=*/false, /*hasSkipCopy=*/false);
      });

  SmallVector<Value> p1LaunchOperands = {
      nttTempDyn,    nttInDyn,      twiddlesDyn,       primeIndicesDyn,
      /*oStart=*/c0, /*iStart=*/c0, primeOffsetVal,
      /*sStart=*/c0, /*sEnd=*/c0,   hasPrimeIndicesVal};
  emitKernelLaunch(b, loc, p1Kernel, gridP1, blockP1, p1LaunchOperands);

  // === Phase 2 Launch: nttTemp -> out ===
  gpu::KernelDim3 gridP2{c128, cDstLen, c1};
  gpu::KernelDim3 blockP2{c64, c1, c1};

  bool hasSrc2 = (op.getSrc2() != nullptr);
  bool hasPadding = (op.getSrc2Padding() != nullptr);
  ModDownVariant variant = ModDownVariant::NoSrc2;
  std::string variantStr = "nosrc2";
  if (hasSrc2 && hasPadding) {
    variant = ModDownVariant::Src2Pad;
    variantStr = "src2_pad";
  } else if (hasSrc2) {
    variant = ModDownVariant::Src2;
    variantStr = "src2";
  }

  std::string p2KernelName =
      llvm::formatv("cheddar_gpu_ntt_fused_fwd_p2_mod_down_{0}_w{1}_deg{2}",
                    variantStr, W, degree)
          .str();

  SmallVector<Type> p2ArgTypes = {dynMemrefType, dynMemrefType, dynMemrefType,
                                  dyn1DMemrefType};
  if (hasSrc2) {
    p2ArgTypes.push_back(dynMemrefType);
  }
  if (hasPadding) {
    p2ArgTypes.push_back(dyn1DMemrefType);
  }
  p2ArgTypes.push_back(dyn1DIndexMemrefType);
  p2ArgTypes.append({indexType, indexType});
  if (hasSrc2) {
    p2ArgTypes.push_back(indexType);
  }
  p2ArgTypes.append({indexType, i1Type});

  FunctionType p2KernelType = FunctionType::get(context, p2ArgTypes, {});
  gpu::GPUFuncOp p2Kernel = getOrCreateKernelWithSharedMem(
      gpuMod, p2KernelName, p2KernelType, /*smemElements=*/512,
      /*smemElemType=*/intType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock, Value smem) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        buildNTTModDownFwdP2Body(ib, entryBlock, smem, W, totalLimbs, variant);
      });

  SmallVector<Value> p2LaunchOperands = {outDyn, nttTempDyn, twiddlesDyn,
                                         invPProdDyn};
  if (hasSrc2) {
    Value src2Dyn = castToDynamicFirstDim(b, loc, op.getSrc2());
    p2LaunchOperands.push_back(src2Dyn);
  }
  if (hasPadding) {
    Value padDyn = castToDynamic1D(b, loc, op.getSrc2Padding());
    p2LaunchOperands.push_back(padDyn);
  }
  p2LaunchOperands.push_back(primeIndicesDyn);
  p2LaunchOperands.append({padStartVal, padEndVal});
  if (hasSrc2) {
    p2LaunchOperands.push_back(s2OffVal);
  }
  p2LaunchOperands.append({primeOffsetVal, hasPrimeIndicesVal});

  emitKernelLaunch(b, loc, p2Kernel, gridP2, blockP2, p2LaunchOperands);
}

LogicalResult decomposeNTTModDown(NTTModDownOp op, PatternRewriter& rewriter) {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto tempType = cast<MemRefType>(op.getNttIn().getType());
  Value nttTemp = memref::AllocOp::create(b, tempType);

  DenseI64ArrayAttr primeRangeAttr;
  if (!op.getPrimeIndices() && op.getPrimeOffset() > 0) {
    primeRangeAttr = b.getDenseI64ArrayAttr(
        {static_cast<int64_t>(op.getPrimeOffset()), tempType.getDimSize(0)});
  }

  NTTOp::create(b, nttTemp, op.getNttIn(),
                /*skip_copy_src=*/Value(),
                FormAttr::get(b.getContext(), Form::Coeff),
                FormAttr::get(b.getContext(), Form::Eval),
                /*montgomery_conversion=*/b.getBoolAttr(false),
                /*skip_range=*/DenseI64ArrayAttr(),
                /*in_limb_range=*/DenseI64ArrayAttr(),
                /*out_limb_range=*/DenseI64ArrayAttr(), primeRangeAttr,
                op.getPrimeIndicesAttr(),
                /*skip_copy_in_offset=*/0);

  Value src2 = op.getSrc2();
  if (src2 && op.getSrc2Offset() > 0) {
    int64_t padLen = op.getPadEnd() - op.getPadStart();
    int64_t degree = tempType.getDimSize(1);
    Type intType = tempType.getElementType();
    auto layout = StridedLayoutAttr::get(b.getContext(),
                                         /*offset=*/op.getSrc2Offset() * degree,
                                         /*strides=*/{degree, 1});
    auto subviewType = MemRefType::get({padLen, degree}, intType, layout);
    SmallVector<OpFoldResult> offsets = {b.getIndexAttr(op.getSrc2Offset()),
                                         b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(padLen),
                                       b.getIndexAttr(degree)};
    SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
    src2 = memref::SubViewOp::create(b, subviewType, op.getSrc2(), offsets,
                                     sizes, strides);
  }

  ModDownEpilogueOp::create(b, op.getOut(), nttTemp, op.getInvPrimeProd(), src2,
                            op.getSrc2Padding(), op.getPadStart(),
                            op.getPadEnd(), Form::Eval, Form::Eval);

  rewriter.eraseOp(op);
  return success();
}

struct DecomposeNTTModDownOp : public OpRewritePattern<NTTModDownOp> {
  using OpRewritePattern<NTTModDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTModDownOp op,
                                PatternRewriter& rewriter) const override {
    return decomposeNTTModDown(op, rewriter);
  }
};

struct ConvertNTTModDownOp : public OpRewritePattern<NTTModDownOp> {
  using OpRewritePattern<NTTModDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTModDownOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent ModuleOp");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    int64_t degree = 1LL << params.getLogN();
    if (degree != 65536) {
      return decomposeNTTModDown(op, rewriter);
    }

    if (params.getWord() == 32) {
      rewriteNTTModDownFused<32>(op, rewriter, params);
    } else if (params.getWord() == 64) {
      rewriteNTTModDownFused<64>(op, rewriter, params);
    } else {
      return op.emitOpError("unsupported word size ") << params.getWord();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateNTTFusedToGPUPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertNTTOp, ConvertINTTOp, ConvertNTTModDownOp>(
      patterns.getContext());
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
