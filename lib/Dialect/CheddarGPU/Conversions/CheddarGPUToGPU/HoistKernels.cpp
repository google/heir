#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/HoistKernels.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "llvm/include/llvm/ADT/APInt.h"                  // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/Hashing.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
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
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

constexpr int64_t kBlockDim = 256;

// Helper to get or create a memoized constant memref.global table of i64 in the
// module.
memref::GlobalOp getOrCreateI64TableGlobal(ModuleOp module, StringRef prefix,
                                           ArrayRef<int64_t> values) {
  uint64_t hash = llvm::hash_combine_range(values.begin(), values.end());
  std::string symName = llvm::formatv("{0}_{1:x-}", prefix, hash).str();
  if (auto g = module.lookupSymbol<memref::GlobalOp>(symName)) {
    return g;
  }

  OpBuilder b(&module.getBodyRegion().front(),
              module.getBodyRegion().front().begin());
  Location loc = module.getLoc();
  Type i64Type = b.getI64Type();
  int64_t count = values.size();
  SmallVector<APInt> apValues;
  apValues.reserve(count);
  for (int64_t v : values) {
    apValues.push_back(APInt(64, static_cast<uint64_t>(v)));
  }

  auto tensorType = RankedTensorType::get({count}, i64Type);
  auto denseAttr = DenseElementsAttr::get(tensorType, apValues);
  auto memrefType = MemRefType::get({count}, i64Type);
  return memref::GlobalOp::create(b, loc, symName, b.getStringAttr("private"),
                                  memrefType, denseAttr, /*constant=*/true,
                                  /*alignment=*/IntegerAttr());
}

// === ConvertHoistBSFusedOp ===

struct ConvertHoistBSFusedOp : public OpRewritePattern<HoistBSFusedOp> {
  using OpRewritePattern<HoistBSFusedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HoistBSFusedOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams) {
      return op.emitOpError("missing cheddar_gpu.params on module");
    }

    uint32_t word = modParams->word;
    int64_t degree = modParams->degree;
    uint32_t logN = modParams->logN;
    size_t R = op.getOutBx().size();
    size_t beta = op.getModUp().size();

    auto outType = ::llvm::cast<MemRefType>(op.getOutBx()[0].getType());
    int64_t numLimbs = outType.getDimSize(0);
    int64_t numQ = op.getNumQ();
    int64_t numAux = op.getNumAux();
    int64_t primeOffset = op.getPrimeOffset();
    int64_t totalPrimes = modParams->params.getNumLimbs();
    int64_t alpha = modParams->params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    Type intType = IntegerType::get(b.getContext(), word);
    Type i64Type = b.getI64Type();
    Type indexType = b.getIndexType();
    auto dynMemref2DType =
        MemRefType::get({ShapedType::kDynamic, degree}, intType);
    auto dynMemref1DWordType = MemRefType::get({ShapedType::kDynamic}, intType);
    auto dynMemref1DI64Type = MemRefType::get({ShapedType::kDynamic}, i64Type);

    // Build the packed params table:
    // [0 .. R-1]: galois_factors
    // [R .. 2R-1]: evk_num_q
    // [2R .. 2R + 2*beta*R - 1]: evk word offsets
    SmallVector<int64_t> paramsTable;
    ArrayRef<int64_t> galois = op.getGaloisFactors();
    ArrayRef<int64_t> evkNumQ = op.getEvkNumQ();
    ArrayRef<int64_t> evkByteOffsets = op.getEvkByteOffsets();
    int64_t wordBytes = word / 8;

    for (size_t k = 0; k < R; ++k) {
      paramsTable.push_back(galois[k]);
    }
    for (size_t k = 0; k < R; ++k) {
      paramsTable.push_back(evkNumQ[k]);
    }

    if (evkByteOffsets.size() == R) {
      for (size_t k = 0; k < R; ++k) {
        int64_t baseWord = evkByteOffsets[k] / wordBytes;
        int64_t lk = evkNumQ[k] + numAux;
        int64_t polyStride = lk * degree;
        for (size_t j = 0; j < beta; ++j) {
          paramsTable.push_back(baseWord + (j * 2 + 0) * polyStride);
          paramsTable.push_back(baseWord + (j * 2 + 1) * polyStride);
        }
      }
    } else {
      for (size_t k = 0; k < R; ++k) {
        for (size_t j = 0; j < beta; ++j) {
          size_t bxIdx = (k * beta + j) * 2 + 0;
          size_t axIdx = (k * beta + j) * 2 + 1;
          paramsTable.push_back(evkByteOffsets[bxIdx] / wordBytes);
          paramsTable.push_back(evkByteOffsets[axIdx] / wordBytes);
        }
      }
    }

    memref::GlobalOp tableGlobal = getOrCreateI64TableGlobal(
        module, "__cheddar_hoist_bs_params", paramsTable);
    Value tableVal = memref::GetGlobalOp::create(b, tableGlobal.getType(),
                                                 tableGlobal.getName());

    // View evk_slab as 1D memref of words
    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value sizeBytes = memref::DimOp::create(b, op.getEvkSlab(), c0);
    Value wordBytesVal = arith::ConstantIndexOp::create(b, wordBytes);
    Value sizeWords = arith::DivUIOp::create(b, sizeBytes, wordBytesVal);
    Value evkSlabWords = memref::ViewOp::create(
        b, dynMemref1DWordType, op.getEvkSlab(), c0, ValueRange{sizeWords});

    // Kernel signature
    std::string kernelName =
        llvm::formatv("cheddar_gpu_hoist_bs_fused_w{0}_deg{1}_k{2}_r{3}_L{4}",
                      word, degree, beta, R, numLimbs)
            .str();

    SmallVector<Type> kernelArgTypes;
    for (size_t i = 0; i < R; ++i) kernelArgTypes.push_back(dynMemref2DType);
    for (size_t i = 0; i < R; ++i) kernelArgTypes.push_back(dynMemref2DType);
    for (size_t j = 0; j < beta; ++j) kernelArgTypes.push_back(dynMemref2DType);
    kernelArgTypes.push_back(dynMemref1DWordType);
    kernelArgTypes.push_back(dynMemref2DType);
    kernelArgTypes.push_back(dynMemref1DI64Type);
    kernelArgTypes.push_back(indexType);
    kernelArgTypes.push_back(indexType);
    kernelArgTypes.push_back(indexType);

    auto kernelType = FunctionType::get(b.getContext(), kernelArgTypes, {});

    getOrCreateDevicePrimeGlobals(gpuMod, modParams->params, word);

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          unsigned argIdx = 0;
          SmallVector<Value> outBxArgs;
          for (size_t i = 0; i < R; ++i)
            outBxArgs.push_back(entryBlock->getArgument(argIdx++));
          SmallVector<Value> outAxArgs;
          for (size_t i = 0; i < R; ++i)
            outAxArgs.push_back(entryBlock->getArgument(argIdx++));
          SmallVector<Value> modUpArgs;
          for (size_t j = 0; j < beta; ++j)
            modUpArgs.push_back(entryBlock->getArgument(argIdx++));
          Value evkSlabArg = entryBlock->getArgument(argIdx++);
          Value inBxPseudoModupArg = entryBlock->getArgument(argIdx++);
          Value paramsTableArg = entryBlock->getArgument(argIdx++);
          Value numQArg = entryBlock->getArgument(argIdx++);
          Value auxStartArg = entryBlock->getArgument(argIdx++);
          Value primeOffsetArg = entryBlock->getArgument(argIdx++);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value tid = arith::AddIOp::create(
              ib, arith::MulIOp::create(ib, bx, bdim), tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value cTotalElems =
              arith::ConstantIndexOp::create(ib, numLimbs * degree);
          Value inBounds = arith::CmpIOp::create(ib, arith::CmpIPredicate::ult,
                                                 tid, cTotalElems);
          auto ifInBounds =
              scf::IfOp::create(ib, inBounds, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifInBounds.getThenRegion().front());

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalPrimes);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          SmallVector<Value> modUpVals;
          for (size_t j = 0; j < beta; ++j) {
            modUpVals.push_back(memref::LoadOp::create(
                ib, modUpArgs[j], ValueRange{localLimb, coeff}));
          }

          Value isQ = arith::CmpIOp::create(ib, arith::CmpIPredicate::slt,
                                            localLimb, numQArg);
          Value c0Idx = arith::ConstantIndexOp::create(ib, 0);
          Value safeLimb = arith::SelectOp::create(ib, isQ, localLimb, c0Idx);
          Value inBxLoaded = memref::LoadOp::create(
              ib, inBxPseudoModupArg, ValueRange{safeLimb, coeff});
          Value c0Word = arith::ConstantIntOp::create(ib, intType, 0);
          Value inBxVal = arith::SelectOp::create(ib, isQ, inBxLoaded, c0Word);

          uint32_t bits = logN + 1;
          Type i32Ty = ib.getI32Type();
          Value xI32 = arith::IndexCastOp::create(ib, i32Ty, coeff);
          Value xRev = buildBitReverse(ib, xI32, bits);
          Value one32 = arith::ConstantIntOp::create(ib, i32Ty, 1);
          Value xRevPlus1 = arith::AddIOp::create(ib, xRev, one32);
          Value xRevPlus1_64 = arith::ExtUIOp::create(ib, i64Type, xRevPlus1);
          Value maskVal =
              arith::ConstantIntOp::create(ib, i64Type, (1ULL << bits) - 1);
          Value one64 = arith::ConstantIntOp::create(ib, i64Type, 1);

          for (size_t k = 0; k < R; ++k) {
            Value cstK = arith::ConstantIndexOp::create(ib, k);
            Value galoisVal =
                memref::LoadOp::create(ib, paramsTableArg, ValueRange{cstK});
            Value prod64 = arith::MulIOp::create(ib, xRevPlus1_64, galoisVal);
            Value prodMinus1 = arith::SubIOp::create(ib, prod64, one64);
            Value mod2N = arith::AndIOp::create(ib, prodMinus1, maskVal);
            Value mod2N_32 = arith::TruncIOp::create(ib, i32Ty, mod2N);
            Value dst32 = buildBitReverse(ib, mod2N_32, bits);
            Value dstIdx =
                arith::IndexCastOp::create(ib, ib.getIndexType(), dst32);

            Value cstRK = arith::ConstantIndexOp::create(ib, R + k);
            Value evkNumQK =
                memref::LoadOp::create(ib, paramsTableArg, ValueRange{cstRK});
            Value evkNumQK_idx =
                arith::IndexCastOp::create(ib, ib.getIndexType(), evkNumQK);
            Value pKeyLimb = arith::AddIOp::create(ib, evkNumQK_idx, auxOffset);
            Value keyLimb =
                arith::SelectOp::create(ib, isAux, pKeyLimb, localLimb);
            Value keyLimbTimesDegPlusCoeff = arith::AddIOp::create(
                ib, arith::MulIOp::create(ib, keyLimb, cDegree), coeff);

            Value resBx = c0Word;
            Value resAx = c0Word;
            for (size_t j = 0; j < beta; ++j) {
              size_t bxTableIdx = 2 * R + (k * beta + j) * 2 + 0;
              size_t axTableIdx = 2 * R + (k * beta + j) * 2 + 1;
              Value cstBxTable = arith::ConstantIndexOp::create(ib, bxTableIdx);
              Value cstAxTable = arith::ConstantIndexOp::create(ib, axTableIdx);
              Value baseBxWord = memref::LoadOp::create(ib, paramsTableArg,
                                                        ValueRange{cstBxTable});
              Value baseAxWord = memref::LoadOp::create(ib, paramsTableArg,
                                                        ValueRange{cstAxTable});
              Value baseBxIdx =
                  arith::IndexCastOp::create(ib, ib.getIndexType(), baseBxWord);
              Value baseAxIdx =
                  arith::IndexCastOp::create(ib, ib.getIndexType(), baseAxWord);

              Value keyBxOffset = arith::AddIOp::create(
                  ib, baseBxIdx, keyLimbTimesDegPlusCoeff);
              Value keyAxOffset = arith::AddIOp::create(
                  ib, baseAxIdx, keyLimbTimesDegPlusCoeff);

              Value keyBxVal = memref::LoadOp::create(ib, evkSlabArg,
                                                      ValueRange{keyBxOffset});
              Value keyAxVal = memref::LoadOp::create(ib, evkSlabArg,
                                                      ValueRange{keyAxOffset});

              Value multBx =
                  buildMontMulRuntime(ib, modUpVals[j], keyBxVal, q, qInv);
              resBx = buildModAddRuntime(ib, resBx, multBx, q);

              Value multAx =
                  buildMontMulRuntime(ib, modUpVals[j], keyAxVal, q, qInv);
              resAx = buildModAddRuntime(ib, resAx, multAx, q);
            }

            Value resBxWithIn = buildModAddRuntime(ib, resBx, inBxVal, q);
            Value finalResBx =
                arith::SelectOp::create(ib, isAux, resBx, resBxWithIn);

            memref::StoreOp::create(ib, finalResBx, outBxArgs[k],
                                    ValueRange{localLimb, dstIdx});
            memref::StoreOp::create(ib, resAx, outAxArgs[k],
                                    ValueRange{localLimb, dstIdx});
          }
        });

    int64_t totalThreads = numLimbs * degree;
    int64_t gridX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value gridDimX = arith::ConstantIndexOp::create(b, gridX);
    Value gridDimY = arith::ConstantIndexOp::create(b, 1);
    Value gridDimZ = arith::ConstantIndexOp::create(b, 1);
    Value blockDimX = arith::ConstantIndexOp::create(b, kBlockDim);
    Value blockDimY = arith::ConstantIndexOp::create(b, 1);
    Value blockDimZ = arith::ConstantIndexOp::create(b, 1);

    Value numQVal = arith::ConstantIndexOp::create(b, numQ);
    Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, primeOffset);

    SmallVector<Value> kernelArgs;
    for (Value v : op.getOutBx())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    for (Value v : op.getOutAx())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    for (Value v : op.getModUp())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    kernelArgs.push_back(evkSlabWords);
    kernelArgs.push_back(
        castToDynamicFirstDim(b, op.getLoc(), op.getInBxPseudoModup()));
    kernelArgs.push_back(castToDynamic1D(b, op.getLoc(), tableVal));
    kernelArgs.push_back(numQVal);
    kernelArgs.push_back(auxStartVal);
    kernelArgs.push_back(primeOffsetVal);

    gpu::LaunchFuncOp::create(b, kernelFunc,
                              gpu::KernelDim3{gridDimX, gridDimY, gridDimZ},
                              gpu::KernelDim3{blockDimX, blockDimY, blockDimZ},
                              /*dynamicSharedMemorySize=*/nullptr, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

// === ConvertHoistGSFusedOp ===

struct ConvertHoistGSFusedOp : public OpRewritePattern<HoistGSFusedOp> {
  using OpRewritePattern<HoistGSFusedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HoistGSFusedOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams) {
      return op.emitOpError("missing cheddar_gpu.params on module");
    }

    uint32_t word = modParams->word;
    int64_t degree = modParams->degree;
    size_t G = op.getOutBx().size();
    size_t B = op.getInBx().size();
    int64_t wordBytes = word / 8;

    auto outType = ::llvm::cast<MemRefType>(op.getOutBx()[0].getType());
    int64_t numLimbs = outType.getDimSize(0);
    int64_t numQ = op.getNumQ();
    int64_t primeOffset = op.getPrimeOffset();
    int64_t totalPrimes = modParams->params.getNumLimbs();
    int64_t alpha = modParams->params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    Type intType = IntegerType::get(b.getContext(), word);
    Type i64Type = b.getI64Type();
    Type indexType = b.getIndexType();
    auto dynMemref2DType =
        MemRefType::get({ShapedType::kDynamic, degree}, intType);
    auto dynMemref1DWordType = MemRefType::get({ShapedType::kDynamic}, intType);
    auto dynMemref1DI64Type = MemRefType::get({ShapedType::kDynamic}, i64Type);

    ArrayRef<int64_t> ptOffsets = op.getPtWordOffsets();
    memref::GlobalOp tableGlobal = getOrCreateI64TableGlobal(
        module, "__cheddar_hoist_gs_pt_offsets", ptOffsets);
    Value tableVal = memref::GetGlobalOp::create(b, tableGlobal.getType(),
                                                 tableGlobal.getName());

    // View consts_slab as 1D memref of words
    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value sizeBytes = memref::DimOp::create(b, op.getConstsSlab(), c0);
    Value wordBytesVal = arith::ConstantIndexOp::create(b, wordBytes);
    Value sizeWords = arith::DivUIOp::create(b, sizeBytes, wordBytesVal);
    Value constsSlabWords = memref::ViewOp::create(
        b, dynMemref1DWordType, op.getConstsSlab(), c0, ValueRange{sizeWords});

    std::string kernelName =
        llvm::formatv("cheddar_gpu_hoist_gs_fused_w{0}_deg{1}_b{2}_g{3}_L{4}",
                      word, degree, B, G, numLimbs)
            .str();

    SmallVector<Type> kernelArgTypes;
    for (size_t g = 0; g < G; ++g) kernelArgTypes.push_back(dynMemref2DType);
    for (size_t g = 0; g < G; ++g) kernelArgTypes.push_back(dynMemref2DType);
    for (size_t b_idx = 0; b_idx < B; ++b_idx)
      kernelArgTypes.push_back(dynMemref2DType);
    for (size_t b_idx = 0; b_idx < B; ++b_idx)
      kernelArgTypes.push_back(dynMemref2DType);
    kernelArgTypes.push_back(dynMemref1DWordType);
    kernelArgTypes.push_back(dynMemref1DI64Type);
    kernelArgTypes.push_back(indexType);
    kernelArgTypes.push_back(indexType);
    kernelArgTypes.push_back(indexType);

    auto kernelType = FunctionType::get(b.getContext(), kernelArgTypes, {});

    getOrCreateDevicePrimeGlobals(gpuMod, modParams->params, word);

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          unsigned argIdx = 0;
          SmallVector<Value> outBxArgs;
          for (size_t g = 0; g < G; ++g)
            outBxArgs.push_back(entryBlock->getArgument(argIdx++));
          SmallVector<Value> outAxArgs;
          for (size_t g = 0; g < G; ++g)
            outAxArgs.push_back(entryBlock->getArgument(argIdx++));
          SmallVector<Value> inBxArgs;
          for (size_t b_idx = 0; b_idx < B; ++b_idx)
            inBxArgs.push_back(entryBlock->getArgument(argIdx++));
          SmallVector<Value> inAxArgs;
          for (size_t b_idx = 0; b_idx < B; ++b_idx)
            inAxArgs.push_back(entryBlock->getArgument(argIdx++));
          Value constsSlabArg = entryBlock->getArgument(argIdx++);
          Value ptOffsetsArg = entryBlock->getArgument(argIdx++);
          Value numQArg = entryBlock->getArgument(argIdx++);
          Value auxStartArg = entryBlock->getArgument(argIdx++);
          Value primeOffsetArg = entryBlock->getArgument(argIdx++);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value tid = arith::AddIOp::create(
              ib, arith::MulIOp::create(ib, bx, bdim), tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value cTotalElems =
              arith::ConstantIndexOp::create(ib, numLimbs * degree);
          Value inBounds = arith::CmpIOp::create(ib, arith::CmpIPredicate::ult,
                                                 tid, cTotalElems);
          auto ifInBounds =
              scf::IfOp::create(ib, inBounds, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifInBounds.getThenRegion().front());

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalPrimes);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          SmallVector<Value> inBxVals;
          SmallVector<Value> inAxVals;
          for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            inBxVals.push_back(memref::LoadOp::create(
                ib, inBxArgs[b_idx], ValueRange{localLimb, coeff}));
            inAxVals.push_back(memref::LoadOp::create(
                ib, inAxArgs[b_idx], ValueRange{localLimb, coeff}));
          }

          Value limbTimesDegPlusCoeff = arith::AddIOp::create(
              ib, arith::MulIOp::create(ib, localLimb, cDegree), coeff);
          Value minusOne = arith::ConstantIntOp::create(ib, i64Type, -1);
          Value c0Word = arith::ConstantIntOp::create(ib, intType, 0);

          for (size_t g = 0; g < G; ++g) {
            Value curResBx = c0Word;
            Value curResAx = c0Word;
            for (size_t b_idx = 0; b_idx < B; ++b_idx) {
              size_t tableIdx = g * B + b_idx;
              Value cstTableIdx = arith::ConstantIndexOp::create(ib, tableIdx);
              Value ptOffsetVal = memref::LoadOp::create(
                  ib, ptOffsetsArg, ValueRange{cstTableIdx});
              Value isPresent = arith::CmpIOp::create(
                  ib, arith::CmpIPredicate::ne, ptOffsetVal, minusOne);

              auto ifOp = scf::IfOp::create(
                  ib, isPresent,
                  [&](OpBuilder& thenB, Location thenLoc) {
                    ImplicitLocOpBuilder tib(thenLoc, thenB);
                    Value ptOffsetIdx = arith::IndexCastOp::create(
                        tib, tib.getIndexType(), ptOffsetVal);
                    Value ptWordIdx = arith::AddIOp::create(
                        tib, ptOffsetIdx, limbTimesDegPlusCoeff);
                    Value ptVal = memref::LoadOp::create(tib, constsSlabArg,
                                                         ValueRange{ptWordIdx});
                    Value multBx = buildMontMulRuntime(tib, inBxVals[b_idx],
                                                       ptVal, q, qInv);
                    Value newResBx =
                        buildModAddRuntime(tib, curResBx, multBx, q);
                    Value multAx = buildMontMulRuntime(tib, inAxVals[b_idx],
                                                       ptVal, q, qInv);
                    Value newResAx =
                        buildModAddRuntime(tib, curResAx, multAx, q);
                    scf::YieldOp::create(tib, ValueRange{newResBx, newResAx});
                  },
                  [&](OpBuilder& elseB, Location elseLoc) {
                    ImplicitLocOpBuilder eib(elseLoc, elseB);
                    scf::YieldOp::create(eib, ValueRange{curResBx, curResAx});
                  });
              curResBx = ifOp.getResult(0);
              curResAx = ifOp.getResult(1);
            }
            memref::StoreOp::create(ib, curResBx, outBxArgs[g],
                                    ValueRange{localLimb, coeff});
            memref::StoreOp::create(ib, curResAx, outAxArgs[g],
                                    ValueRange{localLimb, coeff});
          }
        });

    int64_t totalThreads = numLimbs * degree;
    int64_t gridX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value gridDimX = arith::ConstantIndexOp::create(b, gridX);
    Value gridDimY = arith::ConstantIndexOp::create(b, 1);
    Value gridDimZ = arith::ConstantIndexOp::create(b, 1);
    Value blockDimX = arith::ConstantIndexOp::create(b, kBlockDim);
    Value blockDimY = arith::ConstantIndexOp::create(b, 1);
    Value blockDimZ = arith::ConstantIndexOp::create(b, 1);

    Value numQVal = arith::ConstantIndexOp::create(b, numQ);
    Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, primeOffset);

    SmallVector<Value> kernelArgs;
    for (Value v : op.getOutBx())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    for (Value v : op.getOutAx())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    for (Value v : op.getInBx())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    for (Value v : op.getInAx())
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), v));
    kernelArgs.push_back(constsSlabWords);
    kernelArgs.push_back(castToDynamic1D(b, op.getLoc(), tableVal));
    kernelArgs.push_back(numQVal);
    kernelArgs.push_back(auxStartVal);
    kernelArgs.push_back(primeOffsetVal);

    gpu::LaunchFuncOp::create(b, kernelFunc,
                              gpu::KernelDim3{gridDimX, gridDimY, gridDimZ},
                              gpu::KernelDim3{blockDimX, blockDimY, blockDimZ},
                              /*dynamicSharedMemorySize=*/nullptr, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateHoistToGPUPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertHoistBSFusedOp, ConvertHoistGSFusedOp>(
      patterns.getContext());
}

void configureHoistGPUConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<HoistBSFusedOp, HoistGSFusedOp>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
