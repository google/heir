#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/NTTKernels.h"

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

constexpr int64_t kBlockDim = 256;

template <unsigned W>
void rewriteNTT(NTTOp op, PatternRewriter& rewriter, ParamsAttr paramsAttr,
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
  int64_t halfDegree = degree / 2;
  int64_t logN = paramsAttr.getLogN();
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
  Value lCountVal = b.create<arith::ConstantIndexOp>(loc, lCount);
  Value hasPrimeIndicesVal =
      b.create<arith::ConstantIntOp>(loc, hasPrimeIndices ? 1 : 0, 1);

  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  // 1. Pre-pass: Montgomery conversion or copy
  if (op.getMontgomeryConversion()) {
    std::string r2Name = llvm::formatv("__cheddar_r2_w{0}", W).str();
    auto r2Global = module.lookupSymbol<memref::GlobalOp>(r2Name);
    Value r2Val = getOrCreateHostGetGlobal(b, loc, r2Global, op.getOperation());
    Value r2Dyn = castToDynamic1D(b, loc, r2Val);

    std::string montKernelName =
        llvm::formatv("cheddar_gpu_ntt_fwd_mont_w{0}_deg{1}", W, degree).str();
    FunctionType montKernelType =
        FunctionType::get(context,
                          {dynMemrefType, dynMemrefType, dyn1DMemrefType,
                           dyn1DIndexMemrefType, indexType, indexType,
                           indexType, indexType, indexType, indexType, i1Type},
                          {});

    gpu::GPUFuncOp montKernel = getOrCreateKernel(
        gpuMod, montKernelName, montKernelType,
        [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(kLoc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value r2Arg = entryBlock->getArgument(2);
          Value pIndicesArg = entryBlock->getArgument(3);
          Value oStartArg = entryBlock->getArgument(4);
          Value iStartArg = entryBlock->getArgument(5);
          Value pOffsetArg = entryBlock->getArgument(6);
          Value sStartArg = entryBlock->getArgument(7);
          Value sEndArg = entryBlock->getArgument(8);
          Value lCntArg = entryBlock->getArgument(9);
          Value hIndicesArg = entryBlock->getArgument(10);

          Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
          Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
          Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
          Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
          Value tid = ib.create<arith::AddIOp>(bxDim, tx);

          Value cDeg = ib.create<arith::ConstantIndexOp>(degree);
          Value cDegM1 = ib.create<arith::ConstantIndexOp>(degree - 1);
          Value localLimb = ib.create<arith::DivUIOp>(tid, cDeg);
          Value coeff = ib.create<arith::AndIOp>(tid, cDegM1);

          Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);
          Value inLimb = ib.create<arith::AddIOp>(localLimb, iStartArg);

          Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    localLimb, lCntArg);
          Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                              outLimb, sStartArg);
          Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                              outLimb, sEndArg);
          Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
          Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
          Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);
          Value exec = ib.create<arith::AndIOp>(inBounds, notSkipped);

          auto ifOp = ib.create<scf::IfOp>(exec, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

          Value c0 = ib.create<arith::ConstantIndexOp>(0);
          Value primeOffsetLimb =
              ib.create<arith::AddIOp>(localLimb, pOffsetArg);
          Value safeIdx =
              ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
          Value primeFromIdx =
              ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
          Value absLimb = ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx,
                                                     primeOffsetLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, W, totalLimbs);
          Value q =
              ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
          Value qInv =
              ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});
          Value r2Limb = ib.create<memref::LoadOp>(r2Arg, ValueRange{absLimb});
          Value inVal =
              ib.create<memref::LoadOp>(inArg, ValueRange{inLimb, coeff});
          Value outVal = buildMontMulRuntime(ib, inVal, r2Limb, q, qInv);
          ib.create<memref::StoreOp>(outVal, outArg,
                                     ValueRange{outLimb, coeff});

          ib.setInsertionPointAfter(ifOp);
          ib.create<gpu::ReturnOp>();
        });

    int64_t totalThreads = lCount * degree;
    int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value gridX = b.create<arith::ConstantIndexOp>(loc, gridDimX);
    Value blockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
    emitKernelLaunch(
        b, loc, montKernel, gpu::KernelDim3{gridX, c1, c1},
        gpu::KernelDim3{blockX, c1, c1},
        ValueRange{outDyn, inDyn, r2Dyn, primeIndicesDyn, outStartVal,
                   inStartVal, primeOffsetVal, skipStartVal, skipEndVal,
                   lCountVal, hasPrimeIndicesVal});
  } else if (op.getIn() != op.getOut() || inStart != outStart) {
    std::string copyKernelName =
        llvm::formatv("cheddar_gpu_ntt_fwd_copy_w{0}_deg{1}", W, degree).str();
    FunctionType copyKernelType =
        FunctionType::get(context,
                          {dynMemrefType, dynMemrefType, indexType, indexType,
                           indexType, indexType, indexType},
                          {});

    gpu::GPUFuncOp copyKernel = getOrCreateKernel(
        gpuMod, copyKernelName, copyKernelType,
        [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(kLoc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value oStartArg = entryBlock->getArgument(2);
          Value iStartArg = entryBlock->getArgument(3);
          Value sStartArg = entryBlock->getArgument(4);
          Value sEndArg = entryBlock->getArgument(5);
          Value lCntArg = entryBlock->getArgument(6);

          Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
          Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
          Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
          Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
          Value tid = ib.create<arith::AddIOp>(bxDim, tx);

          Value cDeg = ib.create<arith::ConstantIndexOp>(degree);
          Value cDegM1 = ib.create<arith::ConstantIndexOp>(degree - 1);
          Value localLimb = ib.create<arith::DivUIOp>(tid, cDeg);
          Value coeff = ib.create<arith::AndIOp>(tid, cDegM1);

          Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);
          Value inLimb = ib.create<arith::AddIOp>(localLimb, iStartArg);

          Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    localLimb, lCntArg);
          Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                              outLimb, sStartArg);
          Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                              outLimb, sEndArg);
          Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
          Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
          Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);
          Value exec = ib.create<arith::AndIOp>(inBounds, notSkipped);

          auto ifOp = ib.create<scf::IfOp>(exec, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value inVal =
              ib.create<memref::LoadOp>(inArg, ValueRange{inLimb, coeff});
          ib.create<memref::StoreOp>(inVal, outArg, ValueRange{outLimb, coeff});

          ib.setInsertionPointAfter(ifOp);
          ib.create<gpu::ReturnOp>();
        });

    int64_t totalThreads = lCount * degree;
    int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value gridX = b.create<arith::ConstantIndexOp>(loc, gridDimX);
    Value blockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
    emitKernelLaunch(b, loc, copyKernel, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, inDyn, outStartVal, inStartVal,
                                skipStartVal, skipEndVal, lCountVal});
  }

  if (op.getSkipCopySrc()) {
    int64_t skipStart = (*op.getSkipRange())[0];
    int64_t skipEnd = (*op.getSkipRange())[1];
    int64_t skipCount = skipEnd - skipStart;
    int64_t skipInOff = op.getSkipCopyInOffset();
    Value skipStartVal = b.create<arith::ConstantIndexOp>(loc, skipStart);
    Value skipCountVal = b.create<arith::ConstantIndexOp>(loc, skipCount);
    Value skipInOffVal = b.create<arith::ConstantIndexOp>(loc, skipInOff);
    Value skipSrcDyn = castToDynamicFirstDim(b, loc, op.getSkipCopySrc());
    std::string skipCopyKernelName =
        llvm::formatv("cheddar_gpu_ntt_skip_copy_w{0}_deg{1}", W, degree).str();
    FunctionType skipCopyKernelType = FunctionType::get(
        context,
        {dynMemrefType, dynMemrefType, indexType, indexType, indexType}, {});
    gpu::GPUFuncOp skipCopyKernel = getOrCreateKernel(
        gpuMod, skipCopyKernelName, skipCopyKernelType,
        [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(kLoc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value skipSrcArg = entryBlock->getArgument(1);
          Value sStartArg = entryBlock->getArgument(2);
          Value sCountArg = entryBlock->getArgument(3);
          Value sInOffArg = entryBlock->getArgument(4);

          Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
          Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
          Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
          Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
          Value tid = ib.create<arith::AddIOp>(bxDim, tx);

          Value cDeg = ib.create<arith::ConstantIndexOp>(degree);
          Value cDegM1 = ib.create<arith::ConstantIndexOp>(degree - 1);
          Value localLimb = ib.create<arith::DivUIOp>(tid, cDeg);
          Value coeff = ib.create<arith::AndIOp>(tid, cDegM1);

          Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    localLimb, sCountArg);
          auto ifOp = ib.create<scf::IfOp>(inBounds, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value outLimb = ib.create<arith::AddIOp>(localLimb, sStartArg);
          Value skipLimb = ib.create<arith::AddIOp>(localLimb, sInOffArg);
          Value val = ib.create<memref::LoadOp>(skipSrcArg,
                                                ValueRange{skipLimb, coeff});
          ib.create<memref::StoreOp>(val, outArg, ValueRange{outLimb, coeff});
          ib.setInsertionPointAfter(ifOp);
          ib.create<gpu::ReturnOp>();
        });
    int64_t totalThreads = skipCount * degree;
    int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value gridX = b.create<arith::ConstantIndexOp>(loc, gridDimX);
    Value blockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
    emitKernelLaunch(b, loc, skipCopyKernel, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, skipSrcDyn, skipStartVal, skipCountVal,
                                skipInOffVal});
  }

  // 2. Radix-2 Cooley-Tukey Decimation-in-Time (CT-DIT) butterfly stages
  int64_t totalButterflies = lCount * halfDegree;
  int64_t stageGridDimX = (totalButterflies + kBlockDim - 1) / kBlockDim;
  Value stageGridX = b.create<arith::ConstantIndexOp>(loc, stageGridDimX);
  Value stageBlockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
  gpu::KernelDim3 stageGridSize{stageGridX, c1, c1};
  gpu::KernelDim3 stageBlockSize{stageBlockX, c1, c1};

  FunctionType stageKernelType = FunctionType::get(
      context,
      {dynMemrefType, dynMemrefType, dyn1DIndexMemrefType, indexType, indexType,
       indexType, indexType, indexType, i1Type},
      {});

  for (int64_t stage = 0; stage < logN; ++stage) {
    int64_t m = 1LL << stage;
    int64_t t = halfDegree >> stage;

    std::string stageKernelName =
        llvm::formatv("cheddar_gpu_ntt_fwd_stage{0}_w{1}_deg{2}", stage, W,
                      degree)
            .str();

    gpu::GPUFuncOp stageKernel = getOrCreateKernel(
        gpuMod, stageKernelName, stageKernelType,
        [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(kLoc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value twArg = entryBlock->getArgument(1);
          Value pIndicesArg = entryBlock->getArgument(2);
          Value oStartArg = entryBlock->getArgument(3);
          Value pOffsetArg = entryBlock->getArgument(4);
          Value sStartArg = entryBlock->getArgument(5);
          Value sEndArg = entryBlock->getArgument(6);
          Value lCntArg = entryBlock->getArgument(7);
          Value hIndicesArg = entryBlock->getArgument(8);

          Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
          Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
          Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
          Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
          Value tid = ib.create<arith::AddIOp>(bxDim, tx);

          Value cHalfDeg = ib.create<arith::ConstantIndexOp>(halfDegree);
          Value cHalfDegM1 = ib.create<arith::ConstantIndexOp>(halfDegree - 1);
          Value localLimb = ib.create<arith::DivUIOp>(tid, cHalfDeg);
          Value bflyIdx = ib.create<arith::AndIOp>(tid, cHalfDegM1);

          Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);

          Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    localLimb, lCntArg);
          Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                              outLimb, sStartArg);
          Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                              outLimb, sEndArg);
          Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
          Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
          Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);
          Value exec = ib.create<arith::AndIOp>(inBounds, notSkipped);

          auto ifOp = ib.create<scf::IfOp>(exec, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

          Value c0 = ib.create<arith::ConstantIndexOp>(0);
          Value primeOffsetLimb =
              ib.create<arith::AddIOp>(localLimb, pOffsetArg);
          Value safeIdx =
              ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
          Value primeFromIdx =
              ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
          Value absLimb = ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx,
                                                     primeOffsetLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, W, totalLimbs);
          Value q =
              ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
          Value qInv =
              ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

          Value cT = ib.create<arith::ConstantIndexOp>(t);
          Value c2T = ib.create<arith::ConstantIndexOp>(2 * t);
          Value cM = ib.create<arith::ConstantIndexOp>(m);

          Value i = ib.create<arith::DivUIOp>(bflyIdx, cT);
          Value j;
          if (t > 1) {
            Value cTM1 = ib.create<arith::ConstantIndexOp>(t - 1);
            j = ib.create<arith::AndIOp>(bflyIdx, cTM1);
          } else {
            j = ib.create<arith::ConstantIndexOp>(0);
          }
          Value groupBase = ib.create<arith::MulIOp>(i, c2T);
          Value idx0 = ib.create<arith::AddIOp>(groupBase, j);
          Value idx1 = ib.create<arith::AddIOp>(idx0, cT);
          Value twIdx = ib.create<arith::AddIOp>(cM, i);

          Value w =
              ib.create<memref::LoadOp>(twArg, ValueRange{absLimb, twIdx});
          Value u =
              ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, idx0});
          Value v =
              ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, idx1});

          butterflyNTTRuntime(ib, u, v, w, q, qInv);

          ib.create<memref::StoreOp>(u, outArg, ValueRange{outLimb, idx0});
          ib.create<memref::StoreOp>(v, outArg, ValueRange{outLimb, idx1});

          ib.setInsertionPointAfter(ifOp);
          ib.create<gpu::ReturnOp>();
        });

    emitKernelLaunch(b, loc, stageKernel, stageGridSize, stageBlockSize,
                     ValueRange{outDyn, twiddlesDyn, primeIndicesDyn,
                                outStartVal, primeOffsetVal, skipStartVal,
                                skipEndVal, lCountVal, hasPrimeIndicesVal});
  }
}

template <unsigned W>
void rewriteINTT(INTTOp op, PatternRewriter& rewriter, ParamsAttr paramsAttr,
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
  int64_t halfDegree = degree / 2;
  int64_t logN = paramsAttr.getLogN();
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
  Value lCountVal = b.create<arith::ConstantIndexOp>(loc, lCount);
  Value hasPrimeIndicesVal =
      b.create<arith::ConstantIntOp>(loc, hasPrimeIndices ? 1 : 0, 1);

  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  // 1. Pre-pass copy if in != out or inStart != outStart or in_split is set
  if (op.getIn() != op.getOut() || inStart != outStart || op.getInSplit()) {
    std::string copyKernelName =
        llvm::formatv("cheddar_gpu_intt_fwd_copy_w{0}_deg{1}", W, degree).str();
    FunctionType copyKernelType = FunctionType::get(
        context,
        {dynMemrefType, dynMemrefType, indexType, indexType, indexType,
         indexType, indexType, indexType, indexType},
        {});

    gpu::GPUFuncOp copyKernel = getOrCreateKernel(
        gpuMod, copyKernelName, copyKernelType,
        [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(kLoc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value oStartArg = entryBlock->getArgument(2);
          Value iStartArg = entryBlock->getArgument(3);
          Value sStartArg = entryBlock->getArgument(4);
          Value sEndArg = entryBlock->getArgument(5);
          Value lCntArg = entryBlock->getArgument(6);
          Value inSplitCntArg = entryBlock->getArgument(7);
          Value inExtraLimbsArg = entryBlock->getArgument(8);

          Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
          Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
          Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
          Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
          Value tid = ib.create<arith::AddIOp>(bxDim, tx);

          Value cDeg = ib.create<arith::ConstantIndexOp>(degree);
          Value cDegM1 = ib.create<arith::ConstantIndexOp>(degree - 1);
          Value localLimb = ib.create<arith::DivUIOp>(tid, cDeg);
          Value coeff = ib.create<arith::AndIOp>(tid, cDegM1);

          Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);
          Value c0 = ib.create<arith::ConstantIndexOp>(0);
          Value geSplit = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                                   localLimb, inSplitCntArg);
          Value extraShift =
              ib.create<arith::SelectOp>(geSplit, inExtraLimbsArg, c0);
          Value inLimb = ib.create<arith::AddIOp>(localLimb, iStartArg);
          inLimb = ib.create<arith::AddIOp>(inLimb, extraShift);

          Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    localLimb, lCntArg);
          Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                              outLimb, sStartArg);
          Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                              outLimb, sEndArg);
          Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
          Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
          Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);
          Value exec = ib.create<arith::AndIOp>(inBounds, notSkipped);

          auto ifOp = ib.create<scf::IfOp>(exec, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value inVal =
              ib.create<memref::LoadOp>(inArg, ValueRange{inLimb, coeff});
          ib.create<memref::StoreOp>(inVal, outArg, ValueRange{outLimb, coeff});

          ib.setInsertionPointAfter(ifOp);
          ib.create<gpu::ReturnOp>();
        });

    int64_t totalThreads = lCount * degree;
    int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value gridX = b.create<arith::ConstantIndexOp>(loc, gridDimX);
    Value blockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
    emitKernelLaunch(
        b, loc, copyKernel, gpu::KernelDim3{gridX, c1, c1},
        gpu::KernelDim3{blockX, c1, c1},
        ValueRange{outDyn, inDyn, outStartVal, inStartVal, skipStartVal,
                   skipEndVal, lCountVal, inSplitCntVal, inExtraLimbsVal});
  }

  // 2. Radix-2 Gentleman-Sande Decimation-in-Frequency (GS-DIF) butterfly
  // stages
  int64_t totalButterflies = lCount * halfDegree;
  int64_t stageGridDimX = (totalButterflies + kBlockDim - 1) / kBlockDim;
  Value stageGridX = b.create<arith::ConstantIndexOp>(loc, stageGridDimX);
  Value stageBlockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
  gpu::KernelDim3 stageGridSize{stageGridX, c1, c1};
  gpu::KernelDim3 stageBlockSize{stageBlockX, c1, c1};

  FunctionType stageKernelType = FunctionType::get(
      context,
      {dynMemrefType, dynMemrefType, dyn1DIndexMemrefType, indexType, indexType,
       indexType, indexType, indexType, i1Type},
      {});

  for (int64_t stage = 0; stage < logN; ++stage) {
    int64_t t = 1LL << stage;
    int64_t m = halfDegree >> stage;

    std::string stageKernelName =
        llvm::formatv("cheddar_gpu_intt_inv_stage{0}_w{1}_deg{2}", stage, W,
                      degree)
            .str();

    gpu::GPUFuncOp stageKernel = getOrCreateKernel(
        gpuMod, stageKernelName, stageKernelType,
        [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(kLoc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value twArg = entryBlock->getArgument(1);
          Value pIndicesArg = entryBlock->getArgument(2);
          Value oStartArg = entryBlock->getArgument(3);
          Value pOffsetArg = entryBlock->getArgument(4);
          Value sStartArg = entryBlock->getArgument(5);
          Value sEndArg = entryBlock->getArgument(6);
          Value lCntArg = entryBlock->getArgument(7);
          Value hIndicesArg = entryBlock->getArgument(8);

          Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
          Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
          Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
          Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
          Value tid = ib.create<arith::AddIOp>(bxDim, tx);

          Value cHalfDeg = ib.create<arith::ConstantIndexOp>(halfDegree);
          Value cHalfDegM1 = ib.create<arith::ConstantIndexOp>(halfDegree - 1);
          Value localLimb = ib.create<arith::DivUIOp>(tid, cHalfDeg);
          Value bflyIdx = ib.create<arith::AndIOp>(tid, cHalfDegM1);

          Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);

          Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    localLimb, lCntArg);
          Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                              outLimb, sStartArg);
          Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                              outLimb, sEndArg);
          Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
          Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
          Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);
          Value exec = ib.create<arith::AndIOp>(inBounds, notSkipped);

          auto ifOp = ib.create<scf::IfOp>(exec, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifOp.getThenRegion().front());

          Value c0 = ib.create<arith::ConstantIndexOp>(0);
          Value primeOffsetLimb =
              ib.create<arith::AddIOp>(localLimb, pOffsetArg);
          Value safeIdx =
              ib.create<arith::SelectOp>(hIndicesArg, localLimb, c0);
          Value primeFromIdx =
              ib.create<memref::LoadOp>(pIndicesArg, ValueRange{safeIdx});
          Value absLimb = ib.create<arith::SelectOp>(hIndicesArg, primeFromIdx,
                                                     primeOffsetLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, W, totalLimbs);
          Value q =
              ib.create<memref::LoadOp>(primesMemref, ValueRange{absLimb});
          Value qInv =
              ib.create<memref::LoadOp>(qInvMemref, ValueRange{absLimb});

          Value cT = ib.create<arith::ConstantIndexOp>(t);
          Value c2T = ib.create<arith::ConstantIndexOp>(2 * t);
          Value cM = ib.create<arith::ConstantIndexOp>(m);

          Value i = ib.create<arith::DivUIOp>(bflyIdx, cT);
          Value j;
          if (t > 1) {
            Value cTM1 = ib.create<arith::ConstantIndexOp>(t - 1);
            j = ib.create<arith::AndIOp>(bflyIdx, cTM1);
          } else {
            j = ib.create<arith::ConstantIndexOp>(0);
          }
          Value groupBase = ib.create<arith::MulIOp>(i, c2T);
          Value idx0 = ib.create<arith::AddIOp>(groupBase, j);
          Value idx1 = ib.create<arith::AddIOp>(idx0, cT);
          Value twIdx = ib.create<arith::AddIOp>(cM, i);

          Value w =
              ib.create<memref::LoadOp>(twArg, ValueRange{absLimb, twIdx});
          Value u =
              ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, idx0});
          Value v =
              ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, idx1});

          butterflyINTTRuntime(ib, u, v, w, q, qInv);

          ib.create<memref::StoreOp>(u, outArg, ValueRange{outLimb, idx0});
          ib.create<memref::StoreOp>(v, outArg, ValueRange{outLimb, idx1});

          ib.setInsertionPointAfter(ifOp);
          ib.create<gpu::ReturnOp>();
        });

    emitKernelLaunch(b, loc, stageKernel, stageGridSize, stageBlockSize,
                     ValueRange{outDyn, twiddlesDyn, primeIndicesDyn,
                                outStartVal, primeOffsetVal, skipStartVal,
                                skipEndVal, lCountVal, hasPrimeIndicesVal});
  }

  // 3. Epilogue
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

  std::string epilogueKernelName =
      llvm::formatv("cheddar_gpu_intt_epilogue_{0}_w{1}_deg{2}",
                    epilogueVariant, W, degree)
          .str();

  FunctionType epilogueKernelType = FunctionType::get(
      context,
      {dynMemrefType, dyn1DMemrefType, dyn1DIndexMemrefType, indexType,
       indexType, indexType, indexType, indexType, i1Type},
      {});

  gpu::GPUFuncOp epilogueKernel = getOrCreateKernel(
      gpuMod, epilogueKernelName, epilogueKernelType,
      [&](OpBuilder& kb, Location kLoc, Block* entryBlock) {
        ImplicitLocOpBuilder ib(kLoc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value scalesArg = entryBlock->getArgument(1);
        Value pIndicesArg = entryBlock->getArgument(2);
        Value oStartArg = entryBlock->getArgument(3);
        Value pOffsetArg = entryBlock->getArgument(4);
        Value sStartArg = entryBlock->getArgument(5);
        Value sEndArg = entryBlock->getArgument(6);
        Value lCntArg = entryBlock->getArgument(7);
        Value hIndicesArg = entryBlock->getArgument(8);

        Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
        Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
        Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
        Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
        Value tid = ib.create<arith::AddIOp>(bxDim, tx);

        Value cDeg = ib.create<arith::ConstantIndexOp>(degree);
        Value cDegM1 = ib.create<arith::ConstantIndexOp>(degree - 1);
        Value localLimb = ib.create<arith::DivUIOp>(tid, cDeg);
        Value coeff = ib.create<arith::AndIOp>(tid, cDegM1);

        Value outLimb = ib.create<arith::AddIOp>(localLimb, oStartArg);

        Value inBounds = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                  localLimb, lCntArg);
        Value ge = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge, outLimb,
                                            sStartArg);
        Value lt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt, outLimb,
                                            sEndArg);
        Value isSkipped = ib.create<arith::AndIOp>(ge, lt);
        Value c1B = ib.create<arith::ConstantIntOp>(1, 1);
        Value notSkipped = ib.create<arith::XOrIOp>(isSkipped, c1B);
        Value exec = ib.create<arith::AndIOp>(inBounds, notSkipped);

        auto ifOp = ib.create<scf::IfOp>(exec, /*withElseRegion=*/false);
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

        Value inVal =
            ib.create<memref::LoadOp>(outArg, ValueRange{outLimb, coeff});
        Value scaleVal;
        if (epilogueVariant == "none") {
          scaleVal = ib.create<memref::LoadOp>(scalesArg, ValueRange{absLimb});
        } else {
          scaleVal =
              ib.create<memref::LoadOp>(scalesArg, ValueRange{localLimb});
        }

        Value outVal = buildMontMulRuntime(ib, inVal, scaleVal, q, qInv);
        if (epilogueVariant == "mult_const_normalize") {
          outVal = buildCenteredRuntime(ib, outVal, q);
        }
        ib.create<memref::StoreOp>(outVal, outArg, ValueRange{outLimb, coeff});

        ib.setInsertionPointAfter(ifOp);
        ib.create<gpu::ReturnOp>();
      });

  int64_t totalThreads = lCount * degree;
  int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
  Value gridX = b.create<arith::ConstantIndexOp>(loc, gridDimX);
  Value blockX = b.create<arith::ConstantIndexOp>(loc, kBlockDim);
  emitKernelLaunch(b, loc, epilogueKernel, gpu::KernelDim3{gridX, c1, c1},
                   gpu::KernelDim3{blockX, c1, c1},
                   ValueRange{outDyn, scalesDyn, primeIndicesDyn, outStartVal,
                              primeOffsetVal, skipStartVal, skipEndVal,
                              lCountVal, hasPrimeIndicesVal});
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
      rewriteNTT<32>(op, rewriter, params, inStart, outStart, primeStart,
                     lCount);
    } else if (params.getWord() == 64) {
      rewriteNTT<64>(op, rewriter, params, inStart, outStart, primeStart,
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
      rewriteINTT<32>(op, rewriter, params, inStart, outStart, primeStart,
                      lCount);
    } else if (params.getWord() == 64) {
      rewriteINTT<64>(op, rewriter, params, inStart, outStart, primeStart,
                      lCount);
    } else {
      return op.emitOpError("unsupported word size ") << params.getWord();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct DecomposeNTTModDownOp : public OpRewritePattern<NTTModDownOp> {
  using OpRewritePattern<NTTModDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTModDownOp op,
                                PatternRewriter& rewriter) const override {
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
      auto layout = StridedLayoutAttr::get(
          b.getContext(), /*offset=*/op.getSrc2Offset() * degree,
          /*strides=*/{degree, 1});
      auto subviewType = MemRefType::get({padLen, degree}, intType, layout);
      SmallVector<OpFoldResult> offsets = {b.getIndexAttr(op.getSrc2Offset()),
                                           b.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {b.getIndexAttr(padLen),
                                         b.getIndexAttr(degree)};
      SmallVector<OpFoldResult> strides = {b.getIndexAttr(1),
                                           b.getIndexAttr(1)};
      src2 = memref::SubViewOp::create(b, subviewType, op.getSrc2(), offsets,
                                       sizes, strides);
    }

    ModDownEpilogueOp::create(b, op.getOut(), nttTemp, op.getInvPrimeProd(),
                              src2, op.getSrc2Padding(), op.getPadStart(),
                              op.getPadEnd(), Form::Eval, Form::Eval);

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateNTTToGPUPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertNTTOp, ConvertINTTOp, DecomposeNTTModDownOp>(
      patterns.getContext());
}

void configureNTTGPUConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<NTTOp, INTTOp, NTTModDownOp>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
