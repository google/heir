#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/BConvKernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "llvm/include/llvm/ADT/APInt.h"                  // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"               // from @llvm-project
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

struct ConvertBConvOp : public OpRewritePattern<BConvOp> {
  using OpRewritePattern<BConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BConvOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams) {
      return op.emitOpError("missing cheddar_gpu.params on module");
    }

    uint32_t word = modParams->word;
    uint32_t wideWidth = 2 * word;
    int64_t degree = modParams->degree;

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    auto tableType = cast<MemRefType>(op.getTable().getType());

    auto [outStart, outTotal] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    int64_t tableDstLen = tableType.getDimSize(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    Type intType = IntegerType::get(b.getContext(), word);
    Type wideType = IntegerType::get(b.getContext(), wideWidth);
    Type indexType = b.getIndexType();

    auto dynMemref2DType =
        MemRefType::get({ShapedType::kDynamic, degree}, intType);
    auto dynTable2DType =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, intType);
    auto dynMemref1DType = MemRefType::get({ShapedType::kDynamic}, intType);

    std::optional<ArrayRef<int64_t>> skipRange = op.getSkipRange();
    int64_t skipStart = 0;
    int64_t skipEnd = 0;
    if (skipRange && skipRange->size() == 2) {
      skipStart = (*skipRange)[0];
      skipEnd = (*skipRange)[1];
    }

    int64_t blockSize = std::min<int64_t>(kBlockDim, degree);
    if (blockSize < 1) blockSize = 1;
    int64_t gridDimX = (degree + blockSize - 1) / blockSize;
    int64_t gridDimY = tableDstLen;

    auto dstPrimesAttr = op.getDstPrimes();

    if (!dstPrimesAttr) {
      // Standard BConv: primes looked up from @__cheddar_primes
      std::string kernelName =
          llvm::formatv("cheddar_gpu_bconv_w{0}_deg{1}", word, degree);
      auto kernelType = FunctionType::get(
          b.getContext(),
          {dynMemref2DType, dynMemref2DType, dynTable2DType, indexType,
           indexType, indexType, indexType, indexType},
          {});

      gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
          gpuMod, kernelName, kernelType,
          [&](OpBuilder& kb, Location loc, Block* entryBlock) {
            ImplicitLocOpBuilder ib(loc, kb);
            Value outArg = entryBlock->getArgument(0);
            Value inArg = entryBlock->getArgument(1);
            Value tableArg = entryBlock->getArgument(2);
            Value inStartArg = entryBlock->getArgument(3);
            Value inCountArg = entryBlock->getArgument(4);
            Value outStartArg = entryBlock->getArgument(5);
            Value skipStartArg = entryBlock->getArgument(6);
            Value skipEndArg = entryBlock->getArgument(7);

            Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
            Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
            Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
            Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
            Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
            Value x = ib.create<arith::AddIOp>(bxDim, tx);

            if (degree % blockSize != 0) {
              Value cDegree = ib.create<arith::ConstantIndexOp>(degree);
              Value inBounds = ib.create<arith::CmpIOp>(
                  arith::CmpIPredicate::ult, x, cDegree);
              auto ifOp =
                  ib.create<scf::IfOp>(inBounds, /*withElseRegion=*/false);
              ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
            }

            Value m = by;
            Value hasSkip = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                     skipStartArg, skipEndArg);
            Value isSkip = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                                    m, skipStartArg);
            Value doSkip = ib.create<arith::AndIOp>(hasSkip, isSkip);
            Value skipDiff = ib.create<arith::SubIOp>(skipEndArg, skipStartArg);
            Value dstRelLimb = ib.create<arith::SelectOp>(
                doSkip, ib.create<arith::AddIOp>(m, skipDiff), m);
            Value outLimbIdx =
                ib.create<arith::AddIOp>(outStartArg, dstRelLimb);

            auto [primes, qInvTable] = getDevicePrimeGlobals(ib, word);
            Value q = ib.create<memref::LoadOp>(primes, ValueRange{outLimbIdx});
            Value qInv =
                ib.create<memref::LoadOp>(qInvTable, ValueRange{outLimbIdx});

            Value qWide = ib.create<arith::ExtUIOp>(wideType, q);
            Value cWord = ib.create<arith::ConstantIntOp>(wideType, word);
            Value primeTh = ib.create<arith::ShLIOp>(qWide, cWord);
            Value c1Wide = ib.create<arith::ConstantIntOp>(wideType, 1);
            Value primeThHalf = ib.create<arith::ShRUIOp>(primeTh, c1Wide);
            Value c0Wide = ib.create<arith::ConstantIntOp>(wideType, 0);

            Value c0Idx = ib.create<arith::ConstantIndexOp>(0);
            Value c1Idx = ib.create<arith::ConstantIndexOp>(1);

            auto loop = ib.create<scf::ForOp>(
                c0Idx, inCountArg, c1Idx, ValueRange{c0Wide},
                [&](OpBuilder& lb, Location lLoc, Value sIdx,
                    ValueRange iterArgs) {
                  ImplicitLocOpBuilder lib(lLoc, lb);
                  Value accum = iterArgs[0];
                  Value inLimb = lib.create<arith::AddIOp>(inStartArg, sIdx);
                  Value srcVal =
                      lib.create<memref::LoadOp>(inArg, ValueRange{inLimb, x});
                  Value tblVal =
                      lib.create<memref::LoadOp>(tableArg, ValueRange{m, sIdx});
                  Value srcExt = lib.create<arith::ExtSIOp>(wideType, srcVal);
                  Value tblExt = lib.create<arith::ExtSIOp>(wideType, tblVal);
                  Value prod = lib.create<arith::MulIOp>(srcExt, tblExt);
                  Value newAccum = lib.create<arith::AddIOp>(accum, prod);

                  Value isNeg = lib.create<arith::CmpIOp>(
                      arith::CmpIPredicate::slt, newAccum, c0Wide);
                  Value accumPlus =
                      lib.create<arith::AddIOp>(newAccum, primeTh);
                  Value accumPos =
                      lib.create<arith::SelectOp>(isNeg, accumPlus, newAccum);

                  Value isGeHalf = lib.create<arith::CmpIOp>(
                      arith::CmpIPredicate::sge, accumPos, primeThHalf);
                  Value accumMinus =
                      lib.create<arith::SubIOp>(accumPos, primeTh);
                  Value centered = lib.create<arith::SelectOp>(
                      isGeHalf, accumMinus, accumPos);

                  lib.create<scf::YieldOp>(ValueRange{centered});
                });

            Value finalAccum = loop.getResult(0);
            Value reduced = buildMontReduceRuntime(ib, finalAccum, q, qInv);
            ib.create<memref::StoreOp>(reduced, outArg,
                                       ValueRange{outLimbIdx, x});
            ib.create<gpu::ReturnOp>();
          });

      Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
      Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
      Value tableDyn = castToDynamic2D(b, op.getLoc(), op.getTable());
      Value inStartVal = b.create<arith::ConstantIndexOp>(inStart);
      Value inCountVal = b.create<arith::ConstantIndexOp>(inCount);
      Value outStartVal = b.create<arith::ConstantIndexOp>(outStart);
      Value skipStartVal = b.create<arith::ConstantIndexOp>(skipStart);
      Value skipEndVal = b.create<arith::ConstantIndexOp>(skipEnd);

      Value c1 = b.create<arith::ConstantIndexOp>(1);
      Value gX = b.create<arith::ConstantIndexOp>(gridDimX);
      Value gY = b.create<arith::ConstantIndexOp>(gridDimY);
      Value bX = b.create<arith::ConstantIndexOp>(blockSize);

      emitKernelLaunch(
          b, op.getLoc(), kernelFunc, gpu::KernelDim3{gX, gY, c1},
          gpu::KernelDim3{bX, c1, c1},
          ValueRange{outDyn, inDyn, tableDyn, inStartVal, inCountVal,
                     outStartVal, skipStartVal, skipEndVal});
    } else {
      // Custom destination primes variant
      std::string kernelName = llvm::formatv(
          "cheddar_gpu_bconv_custom_primes_w{0}_deg{1}", word, degree);
      auto kernelType =
          FunctionType::get(b.getContext(),
                            {dynMemref2DType, dynMemref2DType, dynTable2DType,
                             dynMemref1DType, dynMemref1DType, indexType,
                             indexType, indexType, indexType, indexType},
                            {});

      gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
          gpuMod, kernelName, kernelType,
          [&](OpBuilder& kb, Location loc, Block* entryBlock) {
            ImplicitLocOpBuilder ib(loc, kb);
            Value outArg = entryBlock->getArgument(0);
            Value inArg = entryBlock->getArgument(1);
            Value tableArg = entryBlock->getArgument(2);
            Value dstPrimesArg = entryBlock->getArgument(3);
            Value dstQInvArg = entryBlock->getArgument(4);
            Value inStartArg = entryBlock->getArgument(5);
            Value inCountArg = entryBlock->getArgument(6);
            Value outStartArg = entryBlock->getArgument(7);
            Value skipStartArg = entryBlock->getArgument(8);
            Value skipEndArg = entryBlock->getArgument(9);

            Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
            Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
            Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
            Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
            Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
            Value x = ib.create<arith::AddIOp>(bxDim, tx);

            if (degree % blockSize != 0) {
              Value cDegree = ib.create<arith::ConstantIndexOp>(degree);
              Value inBounds = ib.create<arith::CmpIOp>(
                  arith::CmpIPredicate::ult, x, cDegree);
              auto ifOp =
                  ib.create<scf::IfOp>(inBounds, /*withElseRegion=*/false);
              ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
            }

            Value m = by;
            Value hasSkip = ib.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                     skipStartArg, skipEndArg);
            Value isSkip = ib.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                                    m, skipStartArg);
            Value doSkip = ib.create<arith::AndIOp>(hasSkip, isSkip);
            Value skipDiff = ib.create<arith::SubIOp>(skipEndArg, skipStartArg);
            Value dstRelLimb = ib.create<arith::SelectOp>(
                doSkip, ib.create<arith::AddIOp>(m, skipDiff), m);
            Value outLimbIdx =
                ib.create<arith::AddIOp>(outStartArg, dstRelLimb);

            Value q = ib.create<memref::LoadOp>(dstPrimesArg, ValueRange{m});
            Value qInv = ib.create<memref::LoadOp>(dstQInvArg, ValueRange{m});

            Value qWide = ib.create<arith::ExtUIOp>(wideType, q);
            Value cWord = ib.create<arith::ConstantIntOp>(wideType, word);
            Value primeTh = ib.create<arith::ShLIOp>(qWide, cWord);
            Value c1Wide = ib.create<arith::ConstantIntOp>(wideType, 1);
            Value primeThHalf = ib.create<arith::ShRUIOp>(primeTh, c1Wide);
            Value c0Wide = ib.create<arith::ConstantIntOp>(wideType, 0);

            Value c0Idx = ib.create<arith::ConstantIndexOp>(0);
            Value c1Idx = ib.create<arith::ConstantIndexOp>(1);

            auto loop = ib.create<scf::ForOp>(
                c0Idx, inCountArg, c1Idx, ValueRange{c0Wide},
                [&](OpBuilder& lb, Location lLoc, Value sIdx,
                    ValueRange iterArgs) {
                  ImplicitLocOpBuilder lib(lLoc, lb);
                  Value accum = iterArgs[0];
                  Value inLimb = lib.create<arith::AddIOp>(inStartArg, sIdx);
                  Value srcVal =
                      lib.create<memref::LoadOp>(inArg, ValueRange{inLimb, x});
                  Value tblVal =
                      lib.create<memref::LoadOp>(tableArg, ValueRange{m, sIdx});
                  Value srcExt = lib.create<arith::ExtSIOp>(wideType, srcVal);
                  Value tblExt = lib.create<arith::ExtSIOp>(wideType, tblVal);
                  Value prod = lib.create<arith::MulIOp>(srcExt, tblExt);
                  Value newAccum = lib.create<arith::AddIOp>(accum, prod);

                  Value isNeg = lib.create<arith::CmpIOp>(
                      arith::CmpIPredicate::slt, newAccum, c0Wide);
                  Value accumPlus =
                      lib.create<arith::AddIOp>(newAccum, primeTh);
                  Value accumPos =
                      lib.create<arith::SelectOp>(isNeg, accumPlus, newAccum);

                  Value isGeHalf = lib.create<arith::CmpIOp>(
                      arith::CmpIPredicate::sge, accumPos, primeThHalf);
                  Value accumMinus =
                      lib.create<arith::SubIOp>(accumPos, primeTh);
                  Value centered = lib.create<arith::SelectOp>(
                      isGeHalf, accumMinus, accumPos);

                  lib.create<scf::YieldOp>(ValueRange{centered});
                });

            Value finalAccum = loop.getResult(0);
            Value reduced = buildMontReduceRuntime(ib, finalAccum, q, qInv);
            ib.create<memref::StoreOp>(reduced, outArg,
                                       ValueRange{outLimbIdx, x});
            ib.create<gpu::ReturnOp>();
          });

      SmallVector<APInt> primesVec;
      SmallVector<APInt> qInvVec;
      for (int64_t p : *dstPrimesAttr) {
        primesVec.push_back(APInt(word, p));
        uint64_t qInv = (word == 32)
                            ? static_cast<uint64_t>(static_cast<uint32_t>(
                                  cheddar_ref::MontArith<32>::ComputeQInv(p)))
                            : static_cast<uint64_t>(
                                  cheddar_ref::MontArith<64>::ComputeQInv(p));
        qInvVec.push_back(APInt(word, qInv));
      }
      auto tensorType = RankedTensorType::get(
          {static_cast<int64_t>(dstPrimesAttr->size())}, intType);
      auto memrefType = MemRefType::get(
          {static_cast<int64_t>(dstPrimesAttr->size())}, intType);
      auto primesDense = DenseElementsAttr::get(tensorType, primesVec);
      auto qInvDense = DenseElementsAttr::get(tensorType, qInvVec);

      std::string primesGlobalName =
          llvm::formatv("__cheddar_dst_primes_{0}",
                        reinterpret_cast<uintptr_t>(op.getOperation()));
      std::string qInvGlobalName =
          llvm::formatv("__cheddar_dst_qinv_{0}",
                        reinterpret_cast<uintptr_t>(op.getOperation()));

      OpBuilder modB(module.getBodyRegion());
      auto pGlobal = modB.create<memref::GlobalOp>(
          op.getLoc(), primesGlobalName, modB.getStringAttr("private"),
          memrefType, primesDense, /*constant=*/true, IntegerAttr());
      auto qGlobal = modB.create<memref::GlobalOp>(
          op.getLoc(), qInvGlobalName, modB.getStringAttr("private"),
          memrefType, qInvDense, /*constant=*/true, IntegerAttr());

      Value dstPrimesMemref =
          b.create<memref::GetGlobalOp>(memrefType, pGlobal.getName());
      Value dstQInvMemref =
          b.create<memref::GetGlobalOp>(memrefType, qGlobal.getName());

      Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
      Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
      Value tableDyn = castToDynamic2D(b, op.getLoc(), op.getTable());
      Value dstPrimesDyn = castToDynamic1D(b, op.getLoc(), dstPrimesMemref);
      Value dstQInvDyn = castToDynamic1D(b, op.getLoc(), dstQInvMemref);
      Value inStartVal = b.create<arith::ConstantIndexOp>(inStart);
      Value inCountVal = b.create<arith::ConstantIndexOp>(inCount);
      Value outStartVal = b.create<arith::ConstantIndexOp>(outStart);
      Value skipStartVal = b.create<arith::ConstantIndexOp>(skipStart);
      Value skipEndVal = b.create<arith::ConstantIndexOp>(skipEnd);

      Value c1 = b.create<arith::ConstantIndexOp>(1);
      Value gX = b.create<arith::ConstantIndexOp>(gridDimX);
      Value gY = b.create<arith::ConstantIndexOp>(gridDimY);
      Value bX = b.create<arith::ConstantIndexOp>(blockSize);

      emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gX, gY, c1},
                       gpu::KernelDim3{bX, c1, c1},
                       ValueRange{outDyn, inDyn, tableDyn, dstPrimesDyn,
                                  dstQInvDyn, inStartVal, inCountVal,
                                  outStartVal, skipStartVal, skipEndVal});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateBConvToGPUPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertBConvOp>(patterns.getContext());
}

void configureBConvGPUConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<BConvOp>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
