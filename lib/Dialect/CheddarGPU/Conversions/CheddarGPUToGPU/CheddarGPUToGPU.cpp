#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/CheddarGPUToGPU.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/BConvKernels.h"
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/HoistKernels.h"
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/ModDownEpilogueKernels.h"
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/NTTFusedKernels.h"
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/NTTKernels.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMTypes.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_CHEDDARGPUTOGPU
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/CheddarGPUToGPU.h.inc"

namespace {

constexpr int64_t kBlockDim = 256;

// 0. CopyOp: Pointwise buffer or slice copy
struct ConvertCopyOp : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter& rewriter) const override {
    auto inType = cast<MemRefType>(op.getIn().getType());
    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t inTotalLimbs = inType.getDimSize(0);
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = inType.getDimSize(1);
    unsigned word = inType.getElementTypeBitWidth();

    auto [inStart, inCount] = getLimbRange(op.getInLimbRange(), inTotalLimbs);
    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    if (inCount != outCount) {
      return op.emitOpError("copy limb count mismatch");
    }
    if (inCount == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_copy_w{0}_deg{1}", word, degree);

    Type elemType = inType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType = FunctionType::get(
        b.getContext(), {dynMemrefType, dynMemrefType, indexType, indexType},
        {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value outStartArg = entryBlock->getArgument(2);
          Value inStartArg = entryBlock->getArgument(3);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value val =
              memref::LoadOp::create(ib, inArg, ValueRange{inLimb, coeff});
          memref::StoreOp::create(ib, val, outArg, ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
    Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);

    int64_t totalThreads = inCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, inDyn, outStartVal, inStartVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 0b. ZeroPadOp: Zero-fill limb range [start, end)
struct ConvertZeroPadOp : public OpRewritePattern<ZeroPadOp> {
  using OpRewritePattern<ZeroPadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ZeroPadOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;

    auto range = op.getLimbRange();
    int64_t start = range[0];
    int64_t end = range[1];
    int64_t count = end - start;
    if (count <= 0) {
      rewriter.eraseOp(op);
      return success();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_zero_pad_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType = FunctionType::get(
        b.getContext(), {dynMemrefType, indexType, indexType}, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value startLimbArg = entryBlock->getArgument(1);
          Value totalThreadsArg = entryBlock->getArgument(2);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value inBounds = arith::CmpIOp::create(ib, arith::CmpIPredicate::ult,
                                                 tid, totalThreadsArg);
          auto ifInBounds =
              scf::IfOp::create(ib, inBounds, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifInBounds.getThenRegion().front());

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value outLimb = arith::AddIOp::create(ib, localLimb, startLimbArg);
          Value c0 =
              arith::ConstantIntOp::create(ib, ib.getIntegerType(word), 0);
          memref::StoreOp::create(ib, c0, outArg, ValueRange{outLimb, coeff});

          ib.setInsertionPointAfter(ifInBounds);
          gpu::ReturnOp::create(ib);
        });

    Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
    Value startVal = arith::ConstantIndexOp::create(b, start);

    int64_t totalThreads = count * degree;
    int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);
    Value totalThreadsVal = arith::ConstantIndexOp::create(b, totalThreads);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, startVal, totalThreadsVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// Generic binary elementwise helper
template <typename OpTy>
static LogicalResult rewriteBinaryElementwiseOp(
    OpTy op, PatternRewriter& rewriter, StringRef opName,
    llvm::function_ref<Value(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                             Value q, Value qInv)>
        computeFn) {
  auto modParams = getModuleParams(op);
  if (!modParams) return op.emitOpError("missing cheddar_gpu.params on module");

  auto outType = cast<MemRefType>(op.getOut().getType());
  auto lhsType = cast<MemRefType>(op.getLhs().getType());
  auto rhsType = cast<MemRefType>(op.getRhs().getType());
  int64_t outTotalLimbs = outType.getDimSize(0);
  int64_t degree = modParams->degree;
  uint32_t word = modParams->word;
  size_t totalLimbs = modParams->params.getNumLimbs();

  auto [outStart, outCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);
  auto [lhsStart, lhsCount] =
      getLimbRange(op.getLhsLimbRange(), lhsType.getDimSize(0));
  auto [rhsStart, rhsCount] =
      getLimbRange(op.getRhsLimbRange(), rhsType.getDimSize(0));

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->template getParentOfType<ModuleOp>();
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  std::string kernelName =
      llvm::formatv("cheddar_gpu_{0}_w{1}_deg{2}", opName, word, degree);

  Type elemType = outType.getElementType();
  auto dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, elemType);
  Type indexType = b.getIndexType();
  auto kernelType =
      FunctionType::get(b.getContext(),
                        {dynMemrefType, dynMemrefType, dynMemrefType, indexType,
                         indexType, indexType, indexType, indexType, indexType},
                        {});

  gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
      gpuMod, kernelName, kernelType,
      [&](OpBuilder& kb, Location loc, Block* entryBlock) {
        ImplicitLocOpBuilder ib(loc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value lhsArg = entryBlock->getArgument(1);
        Value rhsArg = entryBlock->getArgument(2);
        Value outStartArg = entryBlock->getArgument(3);
        Value lhsStartArg = entryBlock->getArgument(4);
        Value rhsStartArg = entryBlock->getArgument(5);
        Value primeOffsetArg = entryBlock->getArgument(6);
        Value numQArg = entryBlock->getArgument(7);
        Value auxStartArg = entryBlock->getArgument(8);

        Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
        Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
        Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
        Value bxDim = arith::MulIOp::create(ib, bx, bdim);
        Value tid = arith::AddIOp::create(ib, bxDim, tx);

        Value cDegree = arith::ConstantIndexOp::create(ib, degree);
        Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
        Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
        Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

        Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                            localLimb, numQArg);
        Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
        Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
        Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
        Value absLimb =
            arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

        auto [primesMemref, qInvMemref] =
            getDevicePrimeGlobals(ib, word, totalLimbs);
        Value q = memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
        Value qInv =
            memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

        Value lhsLimb = arith::AddIOp::create(ib, localLimb, lhsStartArg);
        Value rhsLimb = arith::AddIOp::create(ib, localLimb, rhsStartArg);
        Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

        Value lhsVal =
            memref::LoadOp::create(ib, lhsArg, ValueRange{lhsLimb, coeff});
        Value rhsVal =
            memref::LoadOp::create(ib, rhsArg, ValueRange{rhsLimb, coeff});

        Value res = computeFn(ib, lhsVal, rhsVal, q, qInv);
        memref::StoreOp::create(ib, res, outArg, ValueRange{outLimb, coeff});
        gpu::ReturnOp::create(ib);
      });

  int64_t numAux = 0;
  if (auto numAuxAttr = op.getNumAux()) numAux = *numAuxAttr;
  int64_t alpha = modParams->params.getAux().size();
  int64_t numQ = outCount - numAux;
  int64_t auxStart = totalLimbs - alpha;

  Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
  Value lhsDyn = castToDynamicFirstDim(b, op.getLoc(), op.getLhs());
  Value rhsDyn = castToDynamicFirstDim(b, op.getLoc(), op.getRhs());
  Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
  Value lhsStartVal = arith::ConstantIndexOp::create(b, lhsStart);
  Value rhsStartVal = arith::ConstantIndexOp::create(b, rhsStart);
  Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);
  Value numQVal = arith::ConstantIndexOp::create(b, numQ);
  Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);

  int64_t totalThreads = outCount * degree;
  int64_t gridDimX = totalThreads / kBlockDim;
  Value c1 = arith::ConstantIndexOp::create(b, 1);
  Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
  Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

  emitKernelLaunch(
      b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
      gpu::KernelDim3{blockX, c1, c1},
      ValueRange{outDyn, lhsDyn, rhsDyn, outStartVal, lhsStartVal, rhsStartVal,
                 primeOffsetVal, numQVal, auxStartVal});

  rewriter.eraseOp(op);
  return success();
}

// 1. AddOp: out = lhs + rhs mod q
struct ConvertAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinaryElementwiseOp<AddOp>(
        op, rewriter, "add",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q,
           Value /*qInv*/) { return buildModAddRuntime(b, lhs, rhs, q); });
  }
};

// 2. SubOp: out = lhs - rhs mod q
struct ConvertSubOp : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinaryElementwiseOp<SubOp>(
        op, rewriter, "sub",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q,
           Value /*qInv*/) { return buildModSubRuntime(b, lhs, rhs, q); });
  }
};

// 3. NegOp: out = -in mod q
struct ConvertNegOp : public OpRewritePattern<NegOp> {
  using OpRewritePattern<NegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_neg_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType =
        FunctionType::get(b.getContext(),
                          {dynMemrefType, dynMemrefType, indexType, indexType,
                           indexType, indexType, indexType},
                          {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value outStartArg = entryBlock->getArgument(2);
          Value inStartArg = entryBlock->getArgument(3);
          Value primeOffsetArg = entryBlock->getArgument(4);
          Value numQArg = entryBlock->getArgument(5);
          Value auxStartArg = entryBlock->getArgument(6);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});

          Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value inVal =
              memref::LoadOp::create(ib, inArg, ValueRange{inLimb, coeff});
          Value res = buildModNegRuntime(ib, inVal, q);
          memref::StoreOp::create(ib, res, outArg, ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    int64_t numAux = 0;
    if (auto numAuxAttr = op.getNumAux()) numAux = *numAuxAttr;
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;

    Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
    Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);
    Value numQVal = arith::ConstantIndexOp::create(b, numQ);
    Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, inDyn, outStartVal, inStartVal,
                                primeOffsetVal, numQVal, auxStartVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// Generic binary 2-poly elementwise helper
template <typename OpTy>
static LogicalResult rewriteBinary2ElementwiseOp(
    OpTy op, PatternRewriter& rewriter, StringRef opName,
    llvm::function_ref<Value(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                             Value q, Value qInv)>
        computeFn) {
  auto modParams = getModuleParams(op);
  if (!modParams) return op.emitOpError("missing cheddar_gpu.params on module");

  auto outType = cast<MemRefType>(op.getOutBx().getType());
  auto lhsType = cast<MemRefType>(op.getLhsBx().getType());
  auto rhsType = cast<MemRefType>(op.getRhsBx().getType());
  int64_t outTotalLimbs = outType.getDimSize(0);
  int64_t degree = modParams->degree;
  uint32_t word = modParams->word;
  size_t totalLimbs = modParams->params.getNumLimbs();

  auto [outStart, outCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);
  auto [lhsStart, lhsCount] =
      getLimbRange(op.getLhsLimbRange(), lhsType.getDimSize(0));
  auto [rhsStart, rhsCount] =
      getLimbRange(op.getRhsLimbRange(), rhsType.getDimSize(0));

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->template getParentOfType<ModuleOp>();
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  std::string kernelName =
      llvm::formatv("cheddar_gpu_{0}_2_w{1}_deg{2}", opName, word, degree);

  Type elemType = outType.getElementType();
  auto dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, elemType);
  Type indexType = b.getIndexType();
  auto kernelType =
      FunctionType::get(b.getContext(),
                        {dynMemrefType, dynMemrefType, dynMemrefType,
                         dynMemrefType, dynMemrefType, dynMemrefType, indexType,
                         indexType, indexType, indexType, indexType, indexType},
                        {});

  gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
      gpuMod, kernelName, kernelType,
      [&](OpBuilder& kb, Location loc, Block* entryBlock) {
        ImplicitLocOpBuilder ib(loc, kb);
        Value outBxArg = entryBlock->getArgument(0);
        Value outAxArg = entryBlock->getArgument(1);
        Value lhsBxArg = entryBlock->getArgument(2);
        Value lhsAxArg = entryBlock->getArgument(3);
        Value rhsBxArg = entryBlock->getArgument(4);
        Value rhsAxArg = entryBlock->getArgument(5);
        Value outStartArg = entryBlock->getArgument(6);
        Value lhsStartArg = entryBlock->getArgument(7);
        Value rhsStartArg = entryBlock->getArgument(8);
        Value primeOffsetArg = entryBlock->getArgument(9);
        Value numQArg = entryBlock->getArgument(10);
        Value auxStartArg = entryBlock->getArgument(11);

        Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
        Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
        Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
        Value bxDim = arith::MulIOp::create(ib, bx, bdim);
        Value tid = arith::AddIOp::create(ib, bxDim, tx);

        Value cDegree = arith::ConstantIndexOp::create(ib, degree);
        Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
        Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
        Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

        Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                            localLimb, numQArg);
        Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
        Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
        Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
        Value absLimb =
            arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

        auto [primesMemref, qInvMemref] =
            getDevicePrimeGlobals(ib, word, totalLimbs);
        Value q = memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
        Value qInv =
            memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

        Value lhsLimb = arith::AddIOp::create(ib, localLimb, lhsStartArg);
        Value rhsLimb = arith::AddIOp::create(ib, localLimb, rhsStartArg);
        Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

        Value lhsBxVal =
            memref::LoadOp::create(ib, lhsBxArg, ValueRange{lhsLimb, coeff});
        Value rhsBxVal =
            memref::LoadOp::create(ib, rhsBxArg, ValueRange{rhsLimb, coeff});
        Value resBx = computeFn(ib, lhsBxVal, rhsBxVal, q, qInv);
        memref::StoreOp::create(ib, resBx, outBxArg,
                                ValueRange{outLimb, coeff});

        Value lhsAxVal =
            memref::LoadOp::create(ib, lhsAxArg, ValueRange{lhsLimb, coeff});
        Value rhsAxVal =
            memref::LoadOp::create(ib, rhsAxArg, ValueRange{rhsLimb, coeff});
        Value resAx = computeFn(ib, lhsAxVal, rhsAxVal, q, qInv);
        memref::StoreOp::create(ib, resAx, outAxArg,
                                ValueRange{outLimb, coeff});

        gpu::ReturnOp::create(ib);
      });

  int64_t numAux = 0;
  if (auto numAuxAttr = op.getNumAux()) numAux = *numAuxAttr;
  int64_t alpha = modParams->params.getAux().size();
  int64_t numQ = outCount - numAux;
  int64_t auxStart = totalLimbs - alpha;

  Value outBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutBx());
  Value outAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutAx());
  Value lhsBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getLhsBx());
  Value lhsAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getLhsAx());
  Value rhsBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getRhsBx());
  Value rhsAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getRhsAx());
  Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
  Value lhsStartVal = arith::ConstantIndexOp::create(b, lhsStart);
  Value rhsStartVal = arith::ConstantIndexOp::create(b, rhsStart);
  Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);
  Value numQVal = arith::ConstantIndexOp::create(b, numQ);
  Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);

  int64_t totalThreads = outCount * degree;
  int64_t gridDimX = totalThreads / kBlockDim;
  Value c1 = arith::ConstantIndexOp::create(b, 1);
  Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
  Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

  emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                   gpu::KernelDim3{blockX, c1, c1},
                   ValueRange{outBxDyn, outAxDyn, lhsBxDyn, lhsAxDyn, rhsBxDyn,
                              rhsAxDyn, outStartVal, lhsStartVal, rhsStartVal,
                              primeOffsetVal, numQVal, auxStartVal});

  rewriter.eraseOp(op);
  return success();
}

// 3b. Add2Op: out_bx = lhs_bx + rhs_bx, out_ax = lhs_ax + rhs_ax
struct ConvertAdd2Op : public OpRewritePattern<Add2Op> {
  using OpRewritePattern<Add2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Add2Op op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinary2ElementwiseOp<Add2Op>(
        op, rewriter, "add",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q,
           Value /*qInv*/) { return buildModAddRuntime(b, lhs, rhs, q); });
  }
};

// 3c. Sub2Op: out_bx = lhs_bx - rhs_bx, out_ax = lhs_ax - rhs_ax
struct ConvertSub2Op : public OpRewritePattern<Sub2Op> {
  using OpRewritePattern<Sub2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Sub2Op op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinary2ElementwiseOp<Sub2Op>(
        op, rewriter, "sub",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q,
           Value /*qInv*/) { return buildModSubRuntime(b, lhs, rhs, q); });
  }
};

// 3d. Neg2Op: out_bx = -in_bx, out_ax = -in_ax
struct ConvertNeg2Op : public OpRewritePattern<Neg2Op> {
  using OpRewritePattern<Neg2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Neg2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto inType = cast<MemRefType>(op.getInBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_neg_2_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType = FunctionType::get(
        b.getContext(),
        {dynMemrefType, dynMemrefType, dynMemrefType, dynMemrefType, indexType,
         indexType, indexType, indexType, indexType},
        {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          Value inBxArg = entryBlock->getArgument(2);
          Value inAxArg = entryBlock->getArgument(3);
          Value outStartArg = entryBlock->getArgument(4);
          Value inStartArg = entryBlock->getArgument(5);
          Value primeOffsetArg = entryBlock->getArgument(6);
          Value numQArg = entryBlock->getArgument(7);
          Value auxStartArg = entryBlock->getArgument(8);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});

          Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value inBxVal =
              memref::LoadOp::create(ib, inBxArg, ValueRange{inLimb, coeff});
          Value resBx = buildModNegRuntime(ib, inBxVal, q);
          memref::StoreOp::create(ib, resBx, outBxArg,
                                  ValueRange{outLimb, coeff});

          Value inAxVal =
              memref::LoadOp::create(ib, inAxArg, ValueRange{inLimb, coeff});
          Value resAx = buildModNegRuntime(ib, inAxVal, q);
          memref::StoreOp::create(ib, resAx, outAxArg,
                                  ValueRange{outLimb, coeff});

          gpu::ReturnOp::create(ib);
        });

    int64_t numAux = 0;
    if (auto numAuxAttr = op.getNumAux()) numAux = *numAuxAttr;
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;

    Value outBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutBx());
    Value outAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutAx());
    Value inBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getInBx());
    Value inAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getInAx());
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);
    Value numQVal = arith::ConstantIndexOp::create(b, numQ);
    Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(
        b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
        gpu::KernelDim3{blockX, c1, c1},
        ValueRange{outBxDyn, outAxDyn, inBxDyn, inAxDyn, outStartVal,
                   inStartVal, primeOffsetVal, numQVal, auxStartVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// Generic const elementwise helper
template <typename OpTy>
static LogicalResult rewriteConstElementwiseOp(
    OpTy op, PatternRewriter& rewriter, StringRef opName,
    llvm::function_ref<Value(ImplicitLocOpBuilder& b, Value inVal, Value cstVal,
                             Value q, Value qInv)>
        computeFn) {
  auto modParams = getModuleParams(op);
  if (!modParams) return op.emitOpError("missing cheddar_gpu.params on module");

  auto outType = cast<MemRefType>(op.getOut().getType());
  auto inType = cast<MemRefType>(op.getIn().getType());
  int64_t outTotalLimbs = outType.getDimSize(0);
  int64_t degree = modParams->degree;
  uint32_t word = modParams->word;
  size_t totalLimbs = modParams->params.getNumLimbs();

  auto [outStart, outCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);
  auto [inStart, inCount] =
      getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

  int64_t primeOffset = 0;
  if constexpr (std::is_same_v<OpTy, MulConstOp>) {
    primeOffset = op.getPrimeOffset().value_or(0);
  }
  int64_t numAux = 0;
  if constexpr (std::is_same_v<OpTy, MulConstOp>) {
    numAux = op.getNumAux().value_or(0);
  }
  int64_t alpha = modParams->params.getAux().size();
  int64_t numQ = outCount - numAux;
  int64_t auxStart = totalLimbs - alpha;

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->template getParentOfType<ModuleOp>();
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  std::string kernelName =
      llvm::formatv("cheddar_gpu_{0}_w{1}_deg{2}", opName, word, degree);

  Type elemType = outType.getElementType();
  auto dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, elemType);
  auto dynCstMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
  Type indexType = b.getIndexType();
  auto kernelType =
      FunctionType::get(b.getContext(),
                        {dynMemrefType, dynMemrefType, dynCstMemrefType,
                         indexType, indexType, indexType, indexType, indexType},
                        {});

  gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
      gpuMod, kernelName, kernelType,
      [&](OpBuilder& kb, Location loc, Block* entryBlock) {
        ImplicitLocOpBuilder ib(loc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value inArg = entryBlock->getArgument(1);
        Value cstArg = entryBlock->getArgument(2);
        Value outStartArg = entryBlock->getArgument(3);
        Value inStartArg = entryBlock->getArgument(4);
        Value primeOffsetArg = entryBlock->getArgument(5);
        Value numQArg = entryBlock->getArgument(6);
        Value auxStartArg = entryBlock->getArgument(7);

        Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
        Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
        Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
        Value bxDim = arith::MulIOp::create(ib, bx, bdim);
        Value tid = arith::AddIOp::create(ib, bxDim, tx);

        Value cDegree = arith::ConstantIndexOp::create(ib, degree);
        Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
        Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
        Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

        Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                            localLimb, numQArg);
        Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
        Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
        Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
        Value absLimb =
            arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

        auto [primesMemref, qInvMemref] =
            getDevicePrimeGlobals(ib, word, totalLimbs);
        Value q = memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
        Value qInv =
            memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

        Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
        Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

        Value inVal =
            memref::LoadOp::create(ib, inArg, ValueRange{inLimb, coeff});
        Value cstVal =
            memref::LoadOp::create(ib, cstArg, ValueRange{localLimb});

        Value res = computeFn(ib, inVal, cstVal, q, qInv);
        memref::StoreOp::create(ib, res, outArg, ValueRange{outLimb, coeff});
        gpu::ReturnOp::create(ib);
      });

  Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
  Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
  Value cstDyn = castToDynamicFirstDim(b, op.getLoc(), op.getCst());
  Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
  Value inStartVal = arith::ConstantIndexOp::create(b, inStart);
  Value primeOffsetVal =
      arith::ConstantIndexOp::create(b, outStart + primeOffset);
  Value numQVal = arith::ConstantIndexOp::create(b, numQ);
  Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);

  int64_t totalThreads = outCount * degree;
  int64_t gridDimX = totalThreads / kBlockDim;
  Value c1 = arith::ConstantIndexOp::create(b, 1);
  Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
  Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

  emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                   gpu::KernelDim3{blockX, c1, c1},
                   ValueRange{outDyn, inDyn, cstDyn, outStartVal, inStartVal,
                              primeOffsetVal, numQVal, auxStartVal});

  rewriter.eraseOp(op);
  return success();
}

// 4. AddConstOp: out = in + cst mod q
struct ConvertAddConstOp : public OpRewritePattern<AddConstOp> {
  using OpRewritePattern<AddConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddConstOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteConstElementwiseOp<AddConstOp>(
        op, rewriter, "add_const",
        [](ImplicitLocOpBuilder& b, Value inVal, Value cstVal, Value q,
           Value /*qInv*/) { return buildModAddRuntime(b, inVal, cstVal, q); });
  }
};

// 5. SubConstOp: out = in - cst mod q
struct ConvertSubConstOp : public OpRewritePattern<SubConstOp> {
  using OpRewritePattern<SubConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubConstOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteConstElementwiseOp<SubConstOp>(
        op, rewriter, "sub_const",
        [](ImplicitLocOpBuilder& b, Value inVal, Value cstVal, Value q,
           Value /*qInv*/) { return buildModSubRuntime(b, inVal, cstVal, q); });
  }
};

// 6. SubOppositeConstOp: out = cst - in mod q
struct ConvertSubOppositeConstOp : public OpRewritePattern<SubOppositeConstOp> {
  using OpRewritePattern<SubOppositeConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOppositeConstOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteConstElementwiseOp<SubOppositeConstOp>(
        op, rewriter, "sub_opposite_const",
        [](ImplicitLocOpBuilder& b, Value inVal, Value cstVal, Value q,
           Value /*qInv*/) { return buildModSubRuntime(b, cstVal, inVal, q); });
  }
};

// 7. MulOp: out = in1 * in2 (Montgomery)
struct ConvertMulOp : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinaryElementwiseOp<MulOp>(
        op, rewriter, "mul",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q, Value qInv) {
          return buildMontMulRuntime(b, lhs, rhs, q, qInv);
        });
  }
};

// 8. MulConstOp: out = in * cst (Montgomery)
struct ConvertMulConstOp : public OpRewritePattern<MulConstOp> {
  using OpRewritePattern<MulConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulConstOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteConstElementwiseOp<MulConstOp>(
        op, rewriter, "mul_const",
        [](ImplicitLocOpBuilder& b, Value inVal, Value cstVal, Value q,
           Value qInv) {
          return buildMontMulRuntime(b, inVal, cstVal, q, qInv);
        });
  }
};

// 9. CPAccumOp: sum_k buildMontMul(ct[k], common[k])
struct ConvertCPAccumOp : public OpRewritePattern<CPAccumOp> {
  using OpRewritePattern<CPAccumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CPAccumOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();
    size_t numSrcs = op.getCtSrcs().size();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(outStart);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName = llvm::formatv(
        "cheddar_gpu_cpaccum_w{0}_deg{1}_k{2}", word, degree, numSrcs);
    for (size_t k = 0; k < numSrcs; ++k) {
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      kernelName += llvm::formatv("_r{0}", commonType.getRank());
    }

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    auto dynCstMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    Type indexType = b.getIndexType();

    SmallVector<Type> argTypes;
    argTypes.push_back(dynMemrefType);  // out
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(dynMemrefType);  // ct[k]
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      argTypes.push_back(commonType.getRank() == 1 ? dynCstMemrefType
                                                   : dynMemrefType);
    }
    argTypes.push_back(indexType);  // out_start
    argTypes.push_back(indexType);  // prime_offset
    argTypes.push_back(indexType);  // num_q
    argTypes.push_back(indexType);  // aux_start
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(indexType);  // ct_offset[k]
      argTypes.push_back(indexType);  // ct_aux_offset[k]
    }

    auto kernelType = FunctionType::get(b.getContext(), argTypes, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          size_t baseIdx = 1 + 2 * numSrcs;
          Value outStartArg = entryBlock->getArgument(baseIdx);
          Value primeOffsetArg = entryBlock->getArgument(baseIdx + 1);
          Value numQArg = entryBlock->getArgument(baseIdx + 2);
          Value auxStartArg = entryBlock->getArgument(baseIdx + 3);
          size_t ctOffsetsBase = baseIdx + 4;

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          Value accVal = nullptr;

          for (size_t k = 0; k < numSrcs; ++k) {
            Value ctArg = entryBlock->getArgument(1 + 2 * k);
            Value commonArg = entryBlock->getArgument(1 + 2 * k + 1);
            auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());

            Value ctOffsetArg = entryBlock->getArgument(ctOffsetsBase + 2 * k);
            Value ctAuxOffsetArg =
                entryBlock->getArgument(ctOffsetsBase + 2 * k + 1);
            Value ctLimbOffset =
                arith::SelectOp::create(ib, isAux, ctAuxOffsetArg, ctOffsetArg);
            Value ctLimb = arith::AddIOp::create(ib, localLimb, ctLimbOffset);

            Value ctVal =
                memref::LoadOp::create(ib, ctArg, ValueRange{ctLimb, coeff});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal =
                  memref::LoadOp::create(ib, commonArg, ValueRange{localLimb});
            } else {
              commonVal = memref::LoadOp::create(ib, commonArg,
                                                 ValueRange{localLimb, coeff});
            }

            Value prod = buildMontMulRuntime(ib, ctVal, commonVal, q, qInv);
            if (!accVal) {
              accVal = prod;
            } else {
              accVal = buildModAddRuntime(ib, accVal, prod, q);
            }
          }

          memref::StoreOp::create(ib, accVal, outArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    SmallVector<Value> kernelArgs;
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOut()));
    for (size_t k = 0; k < numSrcs; ++k) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCtSrcs()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCommonSrcs()[k]));
    }
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, outStart));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, primeOffset));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, numQ));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, auxStart));
    for (size_t k = 0; k < numSrcs; ++k) {
      int64_t ctLimbs =
          cast<MemRefType>(op.getCtSrcs()[k].getType()).getDimSize(0);
      bool isCtFull = (ctLimbs > outTotalLimbs);
      int64_t ctOffset = isCtFull ? primeOffset : 0;
      int64_t ctNumQ = ctLimbs - numAux;
      int64_t ctAuxOffset = isCtFull ? (ctNumQ - numQ) : 0;
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctOffset));
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctAuxOffset));
    }

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1}, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

// 10. CPAccumAddOp: acc + sum_k buildMontMul(ct[k], common[k])
struct ConvertCPAccumAddOp : public OpRewritePattern<CPAccumAddOp> {
  using OpRewritePattern<CPAccumAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CPAccumAddOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();
    size_t numSrcs = op.getCtSrcs().size();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(outStart);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName = llvm::formatv(
        "cheddar_gpu_cpaccum_add_w{0}_deg{1}_k{2}", word, degree, numSrcs);
    for (size_t k = 0; k < numSrcs; ++k) {
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      kernelName += llvm::formatv("_r{0}", commonType.getRank());
    }

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    auto dynCstMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    Type indexType = b.getIndexType();

    SmallVector<Type> argTypes;
    argTypes.push_back(dynMemrefType);  // out
    argTypes.push_back(dynMemrefType);  // acc
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(dynMemrefType);  // ct[k]
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      argTypes.push_back(commonType.getRank() == 1 ? dynCstMemrefType
                                                   : dynMemrefType);
    }
    argTypes.push_back(indexType);  // out_start
    argTypes.push_back(indexType);  // prime_offset
    argTypes.push_back(indexType);  // num_q
    argTypes.push_back(indexType);  // aux_start
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(indexType);  // ct_offset[k]
      argTypes.push_back(indexType);  // ct_aux_offset[k]
    }

    auto kernelType = FunctionType::get(b.getContext(), argTypes, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value accArg = entryBlock->getArgument(1);
          size_t baseIdx = 2 + 2 * numSrcs;
          Value outStartArg = entryBlock->getArgument(baseIdx);
          Value primeOffsetArg = entryBlock->getArgument(baseIdx + 1);
          Value numQArg = entryBlock->getArgument(baseIdx + 2);
          Value auxStartArg = entryBlock->getArgument(baseIdx + 3);
          size_t ctOffsetsBase = baseIdx + 4;

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          Value accVal =
              memref::LoadOp::create(ib, accArg, ValueRange{outLimb, coeff});

          for (size_t k = 0; k < numSrcs; ++k) {
            Value ctArg = entryBlock->getArgument(2 + 2 * k);
            Value commonArg = entryBlock->getArgument(2 + 2 * k + 1);
            auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());

            Value ctOffsetArg = entryBlock->getArgument(ctOffsetsBase + 2 * k);
            Value ctAuxOffsetArg =
                entryBlock->getArgument(ctOffsetsBase + 2 * k + 1);
            Value ctLimbOffset =
                arith::SelectOp::create(ib, isAux, ctAuxOffsetArg, ctOffsetArg);
            Value ctLimb = arith::AddIOp::create(ib, localLimb, ctLimbOffset);

            Value ctVal =
                memref::LoadOp::create(ib, ctArg, ValueRange{ctLimb, coeff});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal =
                  memref::LoadOp::create(ib, commonArg, ValueRange{localLimb});
            } else {
              commonVal = memref::LoadOp::create(ib, commonArg,
                                                 ValueRange{localLimb, coeff});
            }

            Value prod = buildMontMulRuntime(ib, ctVal, commonVal, q, qInv);
            accVal = buildModAddRuntime(ib, accVal, prod, q);
          }

          memref::StoreOp::create(ib, accVal, outArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    SmallVector<Value> kernelArgs;
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOut()));
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getAcc()));
    for (size_t k = 0; k < numSrcs; ++k) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCtSrcs()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCommonSrcs()[k]));
    }
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, outStart));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, primeOffset));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, numQ));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, auxStart));
    for (size_t k = 0; k < numSrcs; ++k) {
      int64_t ctLimbs =
          cast<MemRefType>(op.getCtSrcs()[k].getType()).getDimSize(0);
      bool isCtFull = (ctLimbs > outTotalLimbs);
      int64_t ctOffset = isCtFull ? primeOffset : 0;
      int64_t ctNumQ = ctLimbs - numAux;
      int64_t ctAuxOffset = isCtFull ? (ctNumQ - numQ) : 0;
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctOffset));
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctAuxOffset));
    }

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1}, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

// 10b. CPAccum2Op: sum_k buildMontMul({ct_bx[k], ct_ax[k]}, common[k])
struct ConvertCPAccum2Op : public OpRewritePattern<CPAccum2Op> {
  using OpRewritePattern<CPAccum2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(CPAccum2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();
    size_t numSrcs = op.getCommonSrcs().size();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(outStart);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName = llvm::formatv(
        "cheddar_gpu_cpaccum_2_w{0}_deg{1}_k{2}", word, degree, numSrcs);
    for (size_t k = 0; k < numSrcs; ++k) {
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      kernelName += llvm::formatv("_r{0}", commonType.getRank());
    }

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    auto dynCstMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    Type indexType = b.getIndexType();

    SmallVector<Type> argTypes;
    argTypes.push_back(dynMemrefType);  // out_bx
    argTypes.push_back(dynMemrefType);  // out_ax
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(dynMemrefType);  // ct_bx[k]
      argTypes.push_back(dynMemrefType);  // ct_ax[k]
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      argTypes.push_back(commonType.getRank() == 1 ? dynCstMemrefType
                                                   : dynMemrefType);
    }
    argTypes.push_back(indexType);  // out_start
    argTypes.push_back(indexType);  // prime_offset
    argTypes.push_back(indexType);  // num_q
    argTypes.push_back(indexType);  // aux_start
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(indexType);  // ct_offset[k]
      argTypes.push_back(indexType);  // ct_aux_offset[k]
    }

    auto kernelType = FunctionType::get(b.getContext(), argTypes, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          size_t baseIdx = 2 + 3 * numSrcs;
          Value outStartArg = entryBlock->getArgument(baseIdx);
          Value primeOffsetArg = entryBlock->getArgument(baseIdx + 1);
          Value numQArg = entryBlock->getArgument(baseIdx + 2);
          Value auxStartArg = entryBlock->getArgument(baseIdx + 3);
          size_t ctOffsetsBase = baseIdx + 4;

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          Value accBxVal = nullptr;
          Value accAxVal = nullptr;

          for (size_t k = 0; k < numSrcs; ++k) {
            Value ctBxArg = entryBlock->getArgument(2 + 3 * k);
            Value ctAxArg = entryBlock->getArgument(2 + 3 * k + 1);
            Value commonArg = entryBlock->getArgument(2 + 3 * k + 2);
            auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());

            Value ctOffsetArg = entryBlock->getArgument(ctOffsetsBase + 2 * k);
            Value ctAuxOffsetArg =
                entryBlock->getArgument(ctOffsetsBase + 2 * k + 1);
            Value ctLimbOffset =
                arith::SelectOp::create(ib, isAux, ctAuxOffsetArg, ctOffsetArg);
            Value ctLimb = arith::AddIOp::create(ib, localLimb, ctLimbOffset);

            Value ctBxVal =
                memref::LoadOp::create(ib, ctBxArg, ValueRange{ctLimb, coeff});
            Value ctAxVal =
                memref::LoadOp::create(ib, ctAxArg, ValueRange{ctLimb, coeff});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal =
                  memref::LoadOp::create(ib, commonArg, ValueRange{localLimb});
            } else {
              commonVal = memref::LoadOp::create(ib, commonArg,
                                                 ValueRange{localLimb, coeff});
            }

            Value prodBx = buildMontMulRuntime(ib, ctBxVal, commonVal, q, qInv);
            Value prodAx = buildMontMulRuntime(ib, ctAxVal, commonVal, q, qInv);
            if (!accBxVal) {
              accBxVal = prodBx;
              accAxVal = prodAx;
            } else {
              accBxVal = buildModAddRuntime(ib, accBxVal, prodBx, q);
              accAxVal = buildModAddRuntime(ib, accAxVal, prodAx, q);
            }
          }

          memref::StoreOp::create(ib, accBxVal, outBxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, accAxVal, outAxArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    SmallVector<Value> kernelArgs;
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOutBx()));
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOutAx()));
    for (size_t k = 0; k < numSrcs; ++k) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCtBxSrcs()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCtAxSrcs()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCommonSrcs()[k]));
    }
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, outStart));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, primeOffset));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, numQ));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, auxStart));
    for (size_t k = 0; k < numSrcs; ++k) {
      int64_t ctLimbs =
          cast<MemRefType>(op.getCtBxSrcs()[k].getType()).getDimSize(0);
      bool isCtFull = (ctLimbs > outTotalLimbs);
      int64_t ctOffset = isCtFull ? primeOffset : 0;
      int64_t ctNumQ = ctLimbs - numAux;
      int64_t ctAuxOffset = isCtFull ? (ctNumQ - numQ) : 0;
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctOffset));
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctAuxOffset));
    }

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1}, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

// 10c. CPAccumAdd2Op: {acc_bx, acc_ax} + sum_k buildMontMul({ct_bx[k],
// ct_ax[k]}, common[k])
struct ConvertCPAccumAdd2Op : public OpRewritePattern<CPAccumAdd2Op> {
  using OpRewritePattern<CPAccumAdd2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(CPAccumAdd2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();
    size_t numSrcs = op.getCommonSrcs().size();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(outStart);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName = llvm::formatv(
        "cheddar_gpu_cpaccum_add_2_w{0}_deg{1}_k{2}", word, degree, numSrcs);
    for (size_t k = 0; k < numSrcs; ++k) {
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      kernelName += llvm::formatv("_r{0}", commonType.getRank());
    }

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    auto dynCstMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    Type indexType = b.getIndexType();

    SmallVector<Type> argTypes;
    argTypes.push_back(dynMemrefType);  // out_bx
    argTypes.push_back(dynMemrefType);  // out_ax
    argTypes.push_back(dynMemrefType);  // acc_bx
    argTypes.push_back(dynMemrefType);  // acc_ax
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(dynMemrefType);  // ct_bx[k]
      argTypes.push_back(dynMemrefType);  // ct_ax[k]
      auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());
      argTypes.push_back(commonType.getRank() == 1 ? dynCstMemrefType
                                                   : dynMemrefType);
    }
    argTypes.push_back(indexType);  // out_start
    argTypes.push_back(indexType);  // prime_offset
    argTypes.push_back(indexType);  // num_q
    argTypes.push_back(indexType);  // aux_start
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(indexType);  // ct_offset[k]
      argTypes.push_back(indexType);  // ct_aux_offset[k]
    }

    auto kernelType = FunctionType::get(b.getContext(), argTypes, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          Value accBxArg = entryBlock->getArgument(2);
          Value accAxArg = entryBlock->getArgument(3);
          size_t baseIdx = 4 + 3 * numSrcs;
          Value outStartArg = entryBlock->getArgument(baseIdx);
          Value primeOffsetArg = entryBlock->getArgument(baseIdx + 1);
          Value numQArg = entryBlock->getArgument(baseIdx + 2);
          Value auxStartArg = entryBlock->getArgument(baseIdx + 3);
          size_t ctOffsetsBase = baseIdx + 4;

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          Value accBxVal =
              memref::LoadOp::create(ib, accBxArg, ValueRange{outLimb, coeff});
          Value accAxVal =
              memref::LoadOp::create(ib, accAxArg, ValueRange{outLimb, coeff});

          for (size_t k = 0; k < numSrcs; ++k) {
            Value ctBxArg = entryBlock->getArgument(4 + 3 * k);
            Value ctAxArg = entryBlock->getArgument(4 + 3 * k + 1);
            Value commonArg = entryBlock->getArgument(4 + 3 * k + 2);
            auto commonType = cast<MemRefType>(op.getCommonSrcs()[k].getType());

            Value ctOffsetArg = entryBlock->getArgument(ctOffsetsBase + 2 * k);
            Value ctAuxOffsetArg =
                entryBlock->getArgument(ctOffsetsBase + 2 * k + 1);
            Value ctLimbOffset =
                arith::SelectOp::create(ib, isAux, ctAuxOffsetArg, ctOffsetArg);
            Value ctLimb = arith::AddIOp::create(ib, localLimb, ctLimbOffset);

            Value ctBxVal =
                memref::LoadOp::create(ib, ctBxArg, ValueRange{ctLimb, coeff});
            Value ctAxVal =
                memref::LoadOp::create(ib, ctAxArg, ValueRange{ctLimb, coeff});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal =
                  memref::LoadOp::create(ib, commonArg, ValueRange{localLimb});
            } else {
              commonVal = memref::LoadOp::create(ib, commonArg,
                                                 ValueRange{localLimb, coeff});
            }

            Value prodBx = buildMontMulRuntime(ib, ctBxVal, commonVal, q, qInv);
            Value prodAx = buildMontMulRuntime(ib, ctAxVal, commonVal, q, qInv);
            accBxVal = buildModAddRuntime(ib, accBxVal, prodBx, q);
            accAxVal = buildModAddRuntime(ib, accAxVal, prodAx, q);
          }

          memref::StoreOp::create(ib, accBxVal, outBxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, accAxVal, outAxArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    SmallVector<Value> kernelArgs;
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOutBx()));
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOutAx()));
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getAccBx()));
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getAccAx()));
    for (size_t k = 0; k < numSrcs; ++k) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCtBxSrcs()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCtAxSrcs()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getCommonSrcs()[k]));
    }
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, outStart));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, primeOffset));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, numQ));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, auxStart));
    for (size_t k = 0; k < numSrcs; ++k) {
      int64_t ctLimbs =
          cast<MemRefType>(op.getCtBxSrcs()[k].getType()).getDimSize(0);
      bool isCtFull = (ctLimbs > outTotalLimbs);
      int64_t ctOffset = isCtFull ? primeOffset : 0;
      int64_t ctNumQ = ctLimbs - numAux;
      int64_t ctAuxOffset = isCtFull ? (ctNumQ - numQ) : 0;
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctOffset));
      kernelArgs.push_back(arith::ConstantIndexOp::create(b, ctAuxOffset));
    }

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1}, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

// 10d. ModUpToMaxOp: Broadcast and reduce polynomial limb across primes
struct ConvertModUpToMaxOp : public OpRewritePattern<ModUpToMaxOp> {
  using OpRewritePattern<ModUpToMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModUpToMaxOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    if (outCount == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_mod_up_to_max_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType =
        FunctionType::get(b.getContext(),
                          {dynMemrefType, dynMemrefType, indexType, indexType,
                           indexType, indexType, indexType},
                          {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value outStartArg = entryBlock->getArgument(2);
          Value primeOffsetArg = entryBlock->getArgument(3);
          Value numQArg = entryBlock->getArgument(4);
          Value auxStartArg = entryBlock->getArgument(5);
          Value totalThreadsArg = entryBlock->getArgument(6);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value inBounds = arith::CmpIOp::create(ib, arith::CmpIPredicate::ult,
                                                 tid, totalThreadsArg);
          auto ifInBounds =
              scf::IfOp::create(ib, inBounds, /*withElseRegion=*/false);
          ib.setInsertionPointToStart(&ifInBounds.getThenRegion().front());

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});

          Value c0 = arith::ConstantIndexOp::create(ib, 0);
          Value inVal =
              memref::LoadOp::create(ib, inArg, ValueRange{c0, coeff});
          Value res = arith::RemUIOp::create(ib, inVal, q);

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          memref::StoreOp::create(ib, res, outArg, ValueRange{outLimb, coeff});

          ib.setInsertionPointAfter(ifInBounds);
          gpu::ReturnOp::create(ib);
        });

    Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
    Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, primeOffset);
    Value numQVal = arith::ConstantIndexOp::create(b, numQ);
    Value auxStartVal = arith::ConstantIndexOp::create(b, auxStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = (totalThreads + kBlockDim - 1) / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);
    Value totalThreadsVal = arith::ConstantIndexOp::create(b, totalThreads);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, inDyn, outStartVal, primeOffsetVal,
                                numQVal, auxStartVal, totalThreadsVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 11. TensorOp: Karatsuba degree-1 product
struct ConvertTensorOp : public OpRewritePattern<TensorOp> {
  using OpRewritePattern<TensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_tensor_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType = FunctionType::get(
        b.getContext(),
        {dynMemrefType, dynMemrefType, dynMemrefType, dynMemrefType,
         dynMemrefType, dynMemrefType, dynMemrefType, indexType, indexType},
        {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          Value outRxArg = entryBlock->getArgument(2);
          Value b1Arg = entryBlock->getArgument(3);
          Value a1Arg = entryBlock->getArgument(4);
          Value b2Arg = entryBlock->getArgument(5);
          Value a2Arg = entryBlock->getArgument(6);
          Value outStartArg = entryBlock->getArgument(7);
          Value primeOffsetArg = entryBlock->getArgument(8);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value absLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value b1Val =
              memref::LoadOp::create(ib, b1Arg, ValueRange{localLimb, coeff});
          Value a1Val =
              memref::LoadOp::create(ib, a1Arg, ValueRange{localLimb, coeff});
          Value b2Val =
              memref::LoadOp::create(ib, b2Arg, ValueRange{localLimb, coeff});
          Value a2Val =
              memref::LoadOp::create(ib, a2Arg, ValueRange{localLimb, coeff});

          Value outBx = buildMontMulRuntime(ib, b1Val, b2Val, q, qInv);
          Value outRx = buildMontMulRuntime(ib, a1Val, a2Val, q, qInv);
          Value p1 = buildMontMulRuntime(ib, b1Val, a2Val, q, qInv);
          Value p2 = buildMontMulRuntime(ib, a1Val, b2Val, q, qInv);
          Value outAx = buildModAddRuntime(ib, p1, p2, q);

          memref::StoreOp::create(ib, outBx, outBxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, outAx, outAxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, outRx, outRxArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    Value outBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutBx());
    Value outAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutAx());
    Value outRxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutRx());
    Value b1Dyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrc1Bx());
    Value a1Dyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrc1Ax());
    Value b2Dyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrc2Bx());
    Value a2Dyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrc2Ax());
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outBxDyn, outAxDyn, outRxDyn, b1Dyn, a1Dyn,
                                b2Dyn, a2Dyn, outStartVal, primeOffsetVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 12. TensorSquareOp: Degree-1 ciphertext squaring
struct ConvertTensorSquareOp : public OpRewritePattern<TensorSquareOp> {
  using OpRewritePattern<TensorSquareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorSquareOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_tensor_square_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType =
        FunctionType::get(b.getContext(),
                          {dynMemrefType, dynMemrefType, dynMemrefType,
                           dynMemrefType, dynMemrefType, indexType, indexType},
                          {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          Value outRxArg = entryBlock->getArgument(2);
          Value bxArg = entryBlock->getArgument(3);
          Value axArg = entryBlock->getArgument(4);
          Value outStartArg = entryBlock->getArgument(5);
          Value primeOffsetArg = entryBlock->getArgument(6);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value absLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value bxVal =
              memref::LoadOp::create(ib, bxArg, ValueRange{localLimb, coeff});
          Value axVal =
              memref::LoadOp::create(ib, axArg, ValueRange{localLimb, coeff});

          Value outBx = buildMontMulRuntime(ib, bxVal, bxVal, q, qInv);
          Value outRx = buildMontMulRuntime(ib, axVal, axVal, q, qInv);
          Value prod = buildMontMulRuntime(ib, bxVal, axVal, q, qInv);
          Value outAx = buildModAddRuntime(ib, prod, prod, q);

          memref::StoreOp::create(ib, outBx, outBxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, outAx, outAxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, outRx, outRxArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    Value outBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutBx());
    Value outAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutAx());
    Value outRxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutRx());
    Value bxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrcBx());
    Value axDyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrcAx());
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outBxDyn, outAxDyn, outRxDyn, bxDyn, axDyn,
                                outStartVal, primeOffsetVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 13. MulImaginaryUnitOp
struct ConvertMulImaginaryUnitOp : public OpRewritePattern<MulImaginaryUnitOp> {
  using OpRewritePattern<MulImaginaryUnitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulImaginaryUnitOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_mul_iunit_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    auto dynCstMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    Type indexType = b.getIndexType();
    auto kernelType =
        FunctionType::get(b.getContext(),
                          {dynMemrefType, dynMemrefType, dynCstMemrefType,
                           indexType, indexType, indexType},
                          {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value iunitArg = entryBlock->getArgument(2);
          Value outStartArg = entryBlock->getArgument(3);
          Value inStartArg = entryBlock->getArgument(4);
          Value primeOffsetArg = entryBlock->getArgument(5);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value absLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});
          Value qInv =
              memref::LoadOp::create(ib, qInvMemref, ValueRange{absLimb});

          Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value inVal =
              memref::LoadOp::create(ib, inArg, ValueRange{inLimb, coeff});
          Value psiVal =
              memref::LoadOp::create(ib, iunitArg, ValueRange{localLimb});

          Value cHalfDegree = arith::ConstantIndexOp::create(ib, degree / 2);
          Value isSecondHalf = arith::CmpIOp::create(
              ib, arith::CmpIPredicate::sge, coeff, cHalfDegree);
          Value negPsi = buildModNegRuntime(ib, psiVal, q);
          Value factor =
              arith::SelectOp::create(ib, isSecondHalf, negPsi, psiVal);
          Value res = buildMontMulRuntime(ib, inVal, factor, q, qInv);

          memref::StoreOp::create(ib, res, outArg, ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
    Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
    Value iunitDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIUnit());
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);
    Value primeOffsetVal = arith::ConstantIndexOp::create(b, outStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outDyn, inDyn, iunitDyn, outStartVal,
                                inStartVal, primeOffsetVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 14. PermuteOp
struct ConvertPermuteOp : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    uint64_t galois = 0;
    if (op.getConj()) {
      galois = 2 * degree - 1;
    } else if (op.getGalois().has_value()) {
      galois = *op.getGalois();
    } else {
      return op.emitOpError("permute missing galois factor");
    }

    uint32_t bits = modParams->logN + 1;
    uint64_t mask = (1ULL << bits) - 1;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_permute_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    Type i64Type = b.getI64Type();
    auto kernelType = FunctionType::get(
        b.getContext(),
        {dynMemrefType, dynMemrefType, i64Type, indexType, indexType}, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          Value inArg = entryBlock->getArgument(1);
          Value galoisArg = entryBlock->getArgument(2);
          Value outStartArg = entryBlock->getArgument(3);
          Value inStartArg = entryBlock->getArgument(4);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Type i32Ty = ib.getI32Type();
          Type i64Ty = ib.getI64Type();

          Value xI32 = arith::IndexCastOp::create(ib, i32Ty, coeff);
          Value xRev = buildBitReverse(ib, xI32, bits);
          Value one32 = arith::ConstantIntOp::create(ib, i32Ty, 1);
          Value xRevPlus1 = arith::AddIOp::create(ib, xRev, one32);

          Value xRevPlus1_64 = arith::ExtUIOp::create(ib, i64Ty, xRevPlus1);
          Value prod64 = arith::MulIOp::create(ib, xRevPlus1_64, galoisArg);
          Value one64 = arith::ConstantIntOp::create(ib, i64Ty, 1);
          Value prodMinus1 = arith::SubIOp::create(ib, prod64, one64);
          Value maskVal = arith::ConstantIntOp::create(ib, i64Ty, mask);
          Value mod2N = arith::AndIOp::create(ib, prodMinus1, maskVal);
          Value mod2N_32 = arith::TruncIOp::create(ib, i32Ty, mod2N);
          Value xSrc32 = buildBitReverse(ib, mod2N_32, bits);
          Value xSrcIdx =
              arith::IndexCastOp::create(ib, ib.getIndexType(), xSrc32);

          Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value val =
              memref::LoadOp::create(ib, inArg, ValueRange{inLimb, xSrcIdx});
          memref::StoreOp::create(ib, val, outArg, ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
    Value inDyn = castToDynamicFirstDim(b, op.getLoc(), op.getIn());
    Value galoisVal = arith::ConstantIntOp::create(b, b.getI64Type(), galois);
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(
        b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
        gpu::KernelDim3{blockX, c1, c1},
        ValueRange{outDyn, inDyn, galoisVal, outStartVal, inStartVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 14b. Permute2Op
struct ConvertPermute2Op : public OpRewritePattern<Permute2Op> {
  using OpRewritePattern<Permute2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Permute2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto inType = cast<MemRefType>(op.getInBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    uint64_t galois = 0;
    if (op.getConj()) {
      galois = 2 * degree - 1;
    } else if (op.getGalois().has_value()) {
      galois = *op.getGalois();
    } else {
      return op.emitOpError("permute_2 missing galois factor");
    }

    uint32_t bits = modParams->logN + 1;
    uint64_t mask = (1ULL << bits) - 1;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_permute_2_w{0}_deg{1}", word, degree);

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    Type i64Type = b.getI64Type();
    auto kernelType =
        FunctionType::get(b.getContext(),
                          {dynMemrefType, dynMemrefType, dynMemrefType,
                           dynMemrefType, i64Type, indexType, indexType},
                          {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          Value inBxArg = entryBlock->getArgument(2);
          Value inAxArg = entryBlock->getArgument(3);
          Value galoisArg = entryBlock->getArgument(4);
          Value outStartArg = entryBlock->getArgument(5);
          Value inStartArg = entryBlock->getArgument(6);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Type i32Ty = ib.getI32Type();
          Type i64Ty = ib.getI64Type();

          Value xI32 = arith::IndexCastOp::create(ib, i32Ty, coeff);
          Value xRev = buildBitReverse(ib, xI32, bits);
          Value one32 = arith::ConstantIntOp::create(ib, i32Ty, 1);
          Value xRevPlus1 = arith::AddIOp::create(ib, xRev, one32);

          Value xRevPlus1_64 = arith::ExtUIOp::create(ib, i64Ty, xRevPlus1);
          Value prod64 = arith::MulIOp::create(ib, xRevPlus1_64, galoisArg);
          Value one64 = arith::ConstantIntOp::create(ib, i64Ty, 1);
          Value prodMinus1 = arith::SubIOp::create(ib, prod64, one64);
          Value maskVal = arith::ConstantIntOp::create(ib, i64Ty, mask);
          Value mod2N = arith::AndIOp::create(ib, prodMinus1, maskVal);
          Value mod2N_32 = arith::TruncIOp::create(ib, i32Ty, mod2N);
          Value xSrc32 = buildBitReverse(ib, mod2N_32, bits);
          Value xSrcIdx =
              arith::IndexCastOp::create(ib, ib.getIndexType(), xSrc32);

          Value inLimb = arith::AddIOp::create(ib, localLimb, inStartArg);
          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);

          Value valBx =
              memref::LoadOp::create(ib, inBxArg, ValueRange{inLimb, xSrcIdx});
          Value valAx =
              memref::LoadOp::create(ib, inAxArg, ValueRange{inLimb, xSrcIdx});
          memref::StoreOp::create(ib, valBx, outBxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, valAx, outAxArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    Value outBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutBx());
    Value outAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOutAx());
    Value inBxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getInBx());
    Value inAxDyn = castToDynamicFirstDim(b, op.getLoc(), op.getInAx());
    Value galoisVal = arith::ConstantIntOp::create(b, b.getI64Type(), galois);
    Value outStartVal = arith::ConstantIndexOp::create(b, outStart);
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1},
                     ValueRange{outBxDyn, outAxDyn, inBxDyn, inAxDyn, galoisVal,
                                outStartVal, inStartVal});

    rewriter.eraseOp(op);
    return success();
  }
};

// 15. PermuteAccumOp
struct ConvertPermuteAccumOp : public OpRewritePattern<PermuteAccumOp> {
  using OpRewritePattern<PermuteAccumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteAccumOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();
    size_t numSrcs = op.getSrcs().size();
    bool hasAcc = (op.getAcc() != nullptr);

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(outStart);

    uint32_t bits = modParams->logN + 1;
    uint64_t mask = (1ULL << bits) - 1;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_permute_accum_w{0}_deg{1}_k{2}{3}", word,
                      degree, numSrcs, hasAcc ? "_acc" : "");

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    Type i64Type = b.getI64Type();

    SmallVector<Type> argTypes;
    argTypes.push_back(dynMemrefType);  // out
    if (hasAcc) {
      argTypes.push_back(dynMemrefType);  // acc
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(dynMemrefType);  // src[k]
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(i64Type);  // galois[k]
    }
    argTypes.push_back(indexType);  // out_start
    argTypes.push_back(indexType);  // prime_offset
    argTypes.push_back(indexType);  // num_q
    argTypes.push_back(indexType);  // aux_start

    auto kernelType = FunctionType::get(b.getContext(), argTypes, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outArg = entryBlock->getArgument(0);
          unsigned nextArgIdx = 1;
          Value accArg =
              hasAcc ? entryBlock->getArgument(nextArgIdx++) : nullptr;
          SmallVector<Value> srcArgs;
          for (size_t k = 0; k < numSrcs; ++k) {
            srcArgs.push_back(entryBlock->getArgument(nextArgIdx++));
          }
          SmallVector<Value> galoisArgs;
          for (size_t k = 0; k < numSrcs; ++k) {
            galoisArgs.push_back(entryBlock->getArgument(nextArgIdx++));
          }
          Value outStartArg = entryBlock->getArgument(nextArgIdx++);
          Value primeOffsetArg = entryBlock->getArgument(nextArgIdx++);
          Value numQArg = entryBlock->getArgument(nextArgIdx++);
          Value auxStartArg = entryBlock->getArgument(nextArgIdx++);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});

          Type i32Ty = ib.getI32Type();
          Type i64Ty = ib.getI64Type();

          Value xI32 = arith::IndexCastOp::create(ib, i32Ty, coeff);
          Value xRev = buildBitReverse(ib, xI32, bits);
          Value one32 = arith::ConstantIntOp::create(ib, i32Ty, 1);
          Value xRevPlus1 = arith::AddIOp::create(ib, xRev, one32);
          Value xRevPlus1_64 = arith::ExtUIOp::create(ib, i64Ty, xRevPlus1);
          Value maskVal = arith::ConstantIntOp::create(ib, i64Ty, mask);
          Value one64 = arith::ConstantIntOp::create(ib, i64Ty, 1);

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          Value accVal = nullptr;
          if (hasAcc) {
            accVal = memref::LoadOp::create(ib, accArg,
                                            ValueRange{localLimb, coeff});
          }

          for (size_t k = 0; k < numSrcs; ++k) {
            Value galoisVal = galoisArgs[k];
            Value prod64 = arith::MulIOp::create(ib, xRevPlus1_64, galoisVal);
            Value prodMinus1 = arith::SubIOp::create(ib, prod64, one64);
            Value mod2N = arith::AndIOp::create(ib, prodMinus1, maskVal);
            Value mod2N_32 = arith::TruncIOp::create(ib, i32Ty, mod2N);
            Value xSrc32 = buildBitReverse(ib, mod2N_32, bits);
            Value xSrcIdx =
                arith::IndexCastOp::create(ib, ib.getIndexType(), xSrc32);

            Value srcVal = memref::LoadOp::create(
                ib, srcArgs[k], ValueRange{localLimb, xSrcIdx});
            if (!accVal) {
              accVal = srcVal;
            } else {
              accVal = buildModAddRuntime(ib, accVal, srcVal, q);
            }
          }

          memref::StoreOp::create(ib, accVal, outArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    SmallVector<Value> kernelArgs;
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOut()));
    if (hasAcc) {
      kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getAcc()));
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getSrcs()[k]));
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      uint64_t gf = op.getGaloisFactors()[k];
      kernelArgs.push_back(arith::ConstantIntOp::create(b, b.getI64Type(), gf));
    }
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, outStart));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, primeOffset));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, numQ));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, auxStart));

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1}, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

// 15b. PermuteAccum2Op
struct ConvertPermuteAccum2Op : public OpRewritePattern<PermuteAccum2Op> {
  using OpRewritePattern<PermuteAccum2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteAccum2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();
    size_t numSrcs = op.getSrcsBx().size();
    bool hasAcc = (op.getAccBx() != nullptr);

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(outStart);

    uint32_t bits = modParams->logN + 1;
    uint64_t mask = (1ULL << bits) - 1;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    std::string kernelName =
        llvm::formatv("cheddar_gpu_permute_accum_2_w{0}_deg{1}_k{2}{3}", word,
                      degree, numSrcs, hasAcc ? "_acc" : "");

    Type elemType = outType.getElementType();
    auto dynMemrefType =
        MemRefType::get({ShapedType::kDynamic, degree}, elemType);
    Type indexType = b.getIndexType();
    Type i64Type = b.getI64Type();

    SmallVector<Type> argTypes;
    argTypes.push_back(dynMemrefType);  // out_bx
    argTypes.push_back(dynMemrefType);  // out_ax
    if (hasAcc) {
      argTypes.push_back(dynMemrefType);  // acc_bx
      argTypes.push_back(dynMemrefType);  // acc_ax
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(dynMemrefType);  // src_bx[k]
      argTypes.push_back(dynMemrefType);  // src_ax[k]
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      argTypes.push_back(i64Type);  // galois[k]
    }
    argTypes.push_back(indexType);  // out_start
    argTypes.push_back(indexType);  // prime_offset
    argTypes.push_back(indexType);  // num_q
    argTypes.push_back(indexType);  // aux_start

    auto kernelType = FunctionType::get(b.getContext(), argTypes, {});

    gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
        gpuMod, kernelName, kernelType,
        [&](OpBuilder& kb, Location loc, Block* entryBlock) {
          ImplicitLocOpBuilder ib(loc, kb);
          Value outBxArg = entryBlock->getArgument(0);
          Value outAxArg = entryBlock->getArgument(1);
          unsigned nextArgIdx = 2;
          Value accBxArg =
              hasAcc ? entryBlock->getArgument(nextArgIdx++) : nullptr;
          Value accAxArg =
              hasAcc ? entryBlock->getArgument(nextArgIdx++) : nullptr;
          SmallVector<Value> srcsBxArgs;
          SmallVector<Value> srcsAxArgs;
          for (size_t k = 0; k < numSrcs; ++k) {
            srcsBxArgs.push_back(entryBlock->getArgument(nextArgIdx++));
            srcsAxArgs.push_back(entryBlock->getArgument(nextArgIdx++));
          }
          SmallVector<Value> galoisArgs;
          for (size_t k = 0; k < numSrcs; ++k) {
            galoisArgs.push_back(entryBlock->getArgument(nextArgIdx++));
          }
          Value outStartArg = entryBlock->getArgument(nextArgIdx++);
          Value primeOffsetArg = entryBlock->getArgument(nextArgIdx++);
          Value numQArg = entryBlock->getArgument(nextArgIdx++);
          Value auxStartArg = entryBlock->getArgument(nextArgIdx++);

          Value bx = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
          Value tx = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
          Value bdim = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
          Value bxDim = arith::MulIOp::create(ib, bx, bdim);
          Value tid = arith::AddIOp::create(ib, bxDim, tx);

          Value cDegree = arith::ConstantIndexOp::create(ib, degree);
          Value cDegreeM1 = arith::ConstantIndexOp::create(ib, degree - 1);
          Value localLimb = arith::DivUIOp::create(ib, tid, cDegree);
          Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

          Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                              localLimb, numQArg);
          Value qAbsLimb = arith::AddIOp::create(ib, localLimb, primeOffsetArg);
          Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
          Value auxAbsLimb = arith::AddIOp::create(ib, auxOffset, auxStartArg);
          Value absLimb =
              arith::SelectOp::create(ib, isAux, auxAbsLimb, qAbsLimb);

          auto [primesMemref, qInvMemref] =
              getDevicePrimeGlobals(ib, word, totalLimbs);
          Value q =
              memref::LoadOp::create(ib, primesMemref, ValueRange{absLimb});

          Type i32Ty = ib.getI32Type();
          Type i64Ty = ib.getI64Type();

          Value xI32 = arith::IndexCastOp::create(ib, i32Ty, coeff);
          Value xRev = buildBitReverse(ib, xI32, bits);
          Value one32 = arith::ConstantIntOp::create(ib, i32Ty, 1);
          Value xRevPlus1 = arith::AddIOp::create(ib, xRev, one32);
          Value xRevPlus1_64 = arith::ExtUIOp::create(ib, i64Ty, xRevPlus1);
          Value maskVal = arith::ConstantIntOp::create(ib, i64Ty, mask);
          Value one64 = arith::ConstantIntOp::create(ib, i64Ty, 1);

          Value outLimb = arith::AddIOp::create(ib, localLimb, outStartArg);
          Value accBxVal = nullptr;
          Value accAxVal = nullptr;
          if (hasAcc) {
            accBxVal = memref::LoadOp::create(ib, accBxArg,
                                              ValueRange{localLimb, coeff});
            accAxVal = memref::LoadOp::create(ib, accAxArg,
                                              ValueRange{localLimb, coeff});
          }

          for (size_t k = 0; k < numSrcs; ++k) {
            Value galoisVal = galoisArgs[k];
            Value prod64 = arith::MulIOp::create(ib, xRevPlus1_64, galoisVal);
            Value prodMinus1 = arith::SubIOp::create(ib, prod64, one64);
            Value mod2N = arith::AndIOp::create(ib, prodMinus1, maskVal);
            Value mod2N_32 = arith::TruncIOp::create(ib, i32Ty, mod2N);
            Value xSrc32 = buildBitReverse(ib, mod2N_32, bits);
            Value xSrcIdx =
                arith::IndexCastOp::create(ib, ib.getIndexType(), xSrc32);

            Value srcBxVal = memref::LoadOp::create(
                ib, srcsBxArgs[k], ValueRange{localLimb, xSrcIdx});
            Value srcAxVal = memref::LoadOp::create(
                ib, srcsAxArgs[k], ValueRange{localLimb, xSrcIdx});
            if (!accBxVal) {
              accBxVal = srcBxVal;
              accAxVal = srcAxVal;
            } else {
              accBxVal = buildModAddRuntime(ib, accBxVal, srcBxVal, q);
              accAxVal = buildModAddRuntime(ib, accAxVal, srcAxVal, q);
            }
          }

          memref::StoreOp::create(ib, accBxVal, outBxArg,
                                  ValueRange{outLimb, coeff});
          memref::StoreOp::create(ib, accAxVal, outAxArg,
                                  ValueRange{outLimb, coeff});
          gpu::ReturnOp::create(ib);
        });

    SmallVector<Value> kernelArgs;
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOutBx()));
    kernelArgs.push_back(castToDynamicFirstDim(b, op.getLoc(), op.getOutAx()));
    if (hasAcc) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getAccBx()));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getAccAx()));
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getSrcsBx()[k]));
      kernelArgs.push_back(
          castToDynamicFirstDim(b, op.getLoc(), op.getSrcsAx()[k]));
    }
    for (size_t k = 0; k < numSrcs; ++k) {
      uint64_t gf = op.getGaloisFactors()[k];
      kernelArgs.push_back(arith::ConstantIntOp::create(b, b.getI64Type(), gf));
    }
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, outStart));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, primeOffset));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, numQ));
    kernelArgs.push_back(arith::ConstantIndexOp::create(b, auxStart));

    int64_t totalThreads = outCount * degree;
    int64_t gridDimX = totalThreads / kBlockDim;
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
    Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

    emitKernelLaunch(b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
                     gpu::KernelDim3{blockX, c1, c1}, kernelArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

static bool isBufferConsumedByGpu(Value rootVal) {
  SmallVector<Value> worklist = {rootVal};
  llvm::DenseSet<Value> visited = {rootVal};

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation* user : val.getUsers()) {
      if (isa<gpu::LaunchOp, gpu::LaunchFuncOp>(user)) return true;
      if (user->getParentOfType<gpu::LaunchOp>() ||
          user->getParentOfType<gpu::GPUFuncOp>())
        return true;
      if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp,
              memref::ViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(
              user)) {
        for (Value res : user->getResults()) {
          if (visited.insert(res).second) worklist.push_back(res);
        }
      } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [initArg, iterArg] :
             llvm::zip(forOp.getInits(), forOp.getRegionIterArgs())) {
          if (initArg == val && visited.insert(iterArg).second) {
            worklist.push_back(iterArg);
          }
        }
        for (auto [initArg, res] :
             llvm::zip(forOp.getInits(), forOp.getResults())) {
          if (initArg == val && visited.insert(res).second) {
            worklist.push_back(res);
          }
        }
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        Operation* parentOp = yieldOp->getParentOp();
        for (OpOperand& operand : yieldOp->getOpOperands()) {
          if (operand.get() == val) {
            unsigned opndNum = operand.getOperandNumber();
            if (auto parentFor = dyn_cast<scf::ForOp>(parentOp)) {
              if (opndNum < parentFor.getNumResults()) {
                Value res = parentFor.getResult(opndNum);
                if (visited.insert(res).second) worklist.push_back(res);
              }
              if (opndNum < parentFor.getRegionIterArgs().size()) {
                Value iterArg = parentFor.getRegionIterArgs()[opndNum];
                if (visited.insert(iterArg).second) worklist.push_back(iterArg);
              }
            } else if (parentOp && opndNum < parentOp->getNumResults()) {
              Value res = parentOp->getResult(opndNum);
              if (visited.insert(res).second) worklist.push_back(res);
            }
          }
        }
      } else if (auto callOp = dyn_cast<func::CallOp>(user)) {
        auto module = user->getParentOfType<ModuleOp>();
        if (module) {
          if (auto callee =
                  module.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
            if (!callee.isExternal() && !callee.empty()) {
              for (OpOperand& operand : callOp->getOpOperands()) {
                if (operand.get() == val) {
                  unsigned argIdx = operand.getOperandNumber();
                  if (argIdx < callee.getNumArguments()) {
                    Value calleeArg = callee.getArgument(argIdx);
                    if (visited.insert(calleeArg).second) {
                      worklist.push_back(calleeArg);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

static void cloneGetGlobalsInsideLaunches(ModuleOp module) {
  module.walk([&](gpu::LaunchOp launchOp) {
    Block& entryBlock = launchOp.getBody().front();
    OpBuilder b(&entryBlock, entryBlock.begin());
    llvm::DenseMap<Value, Value> replacements;
    launchOp.walk([&](Operation* user) {
      if (user == launchOp.getOperation()) return;
      for (OpOperand& operand : user->getOpOperands()) {
        Value val = operand.get();
        Operation* defOp = val.getDefiningOp();
        if (defOp && !launchOp.getOperation()->isAncestor(defOp)) {
          if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defOp)) {
            auto it = replacements.find(val);
            if (it == replacements.end()) {
              auto cloned = memref::GetGlobalOp::create(b, getGlobal.getLoc(),
                                                        getGlobal.getType(),
                                                        getGlobal.getName());
              it = replacements.insert({val, cloned.getResult()}).first;
            }
            operand.set(it->second);
          }
        }
      }
    });
  });
}

static Value reconstructDeviceMemRef(OpBuilder& b, Location loc,
                                     MemRefType memrefType, Value rawPtr) {
  MLIRContext* ctx = b.getContext();
  Type llvmPtrType = LLVM::LLVMPointerType::get(ctx);
  Type i64Type = IntegerType::get(ctx, 64);
  int64_t rank = memrefType.getRank();
  Type sizesArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  Type stridesArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  Type structType = LLVM::LLVMStructType::getLiteral(
      ctx,
      {llvmPtrType, llvmPtrType, i64Type, sizesArrayType, stridesArrayType});

  Value desc = LLVM::UndefOp::create(b, loc, structType);
  desc =
      LLVM::InsertValueOp::create(b, loc, desc, rawPtr, ArrayRef<int64_t>{0});
  desc =
      LLVM::InsertValueOp::create(b, loc, desc, rawPtr, ArrayRef<int64_t>{1});
  Value c0 = LLVM::ConstantOp::create(b, loc, i64Type, b.getI64IntegerAttr(0));
  desc = LLVM::InsertValueOp::create(b, loc, desc, c0, ArrayRef<int64_t>{2});

  SmallVector<int64_t> strides(rank);
  int64_t curStride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = curStride;
    curStride *= memrefType.getDimSize(i);
  }

  for (int64_t d = 0; d < rank; ++d) {
    Value dimSize = LLVM::ConstantOp::create(
        b, loc, i64Type, b.getI64IntegerAttr(memrefType.getDimSize(d)));
    desc = LLVM::InsertValueOp::create(b, loc, desc, dimSize,
                                       ArrayRef<int64_t>{3, d});
    Value strideVal = LLVM::ConstantOp::create(b, loc, i64Type,
                                               b.getI64IntegerAttr(strides[d]));
    desc = LLVM::InsertValueOp::create(b, loc, desc, strideVal,
                                       ArrayRef<int64_t>{4, d});
  }

  return UnrealizedConversionCastOp::create(b, loc, memrefType, desc)
      .getResult(0);
}

static Value extractRawPointerFromDeviceMemRef(OpBuilder& b, Location loc,
                                               Value memrefVal,
                                               MemRefType memrefType) {
  MLIRContext* ctx = b.getContext();
  Type llvmPtrType = LLVM::LLVMPointerType::get(ctx);
  Type i64Type = IntegerType::get(ctx, 64);
  int64_t rank = memrefType.getRank();
  Type sizesArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  Type stridesArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  Type structType = LLVM::LLVMStructType::getLiteral(
      ctx,
      {llvmPtrType, llvmPtrType, i64Type, sizesArrayType, stridesArrayType});
  Value structVal =
      UnrealizedConversionCastOp::create(b, loc, structType, memrefVal)
          .getResult(0);
  return LLVM::ExtractValueOp::create(b, loc, structVal, ArrayRef<int64_t>{0});
}

//===----------------------------------------------------------------------===//
// CUDA Stream Model & Device Residency
//
// 1. Single Stream Model: All generated kernel launches in entry functions
//    execute on the legacy default stream (stream 0). No async token chains
//    (`gpu.wait async`) are created or used in entry functions, ensuring that
//    all kernel launches execute in strict FIFO order on stream 0 without
//    inter-stream race conditions.
//
// 2. One-Time Initialization (__cheddar_gpu_init):
//    On the first call to any function in the module, `@__cheddar_gpu_init`
//    allocates the persistent scratch arena and copies host constant tables
//    to device memory.
//    Crucially, the initialization branch ends with a host-blocking wait
//    (`gpu.wait %currentToken` with no result) and `cudaDeviceSynchronize()`.
//    This guarantees that all device tables and arena memory are fully
//    allocated, copied, and coherent before `@__cheddar_gpu_init` returns.
//
// 3. Zero Steady-State Overhead:
//    Subsequent calls to `@__cheddar_gpu_init` check `@__cheddar_gpu_init_flag`
//    and return immediately. Entry functions contain NO per-call
//    `cudaDeviceSynchronize` or `gpu.dealloc` calls, achieving steady-state
//    host latency <= ~0.3 ms over pure kernel execution time.
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "cheddar-gpu-arena"

struct FreeBlock {
  int64_t offset;
  int64_t size;
};

class FirstFitFreeList {
 public:
  FirstFitFreeList() {
    blocks.push_back({0, std::numeric_limits<int64_t>::max() / 2});
  }

  int64_t allocate(int64_t sizeBytes) {
    assert(sizeBytes % 256 == 0);
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
      if (it->size >= sizeBytes) {
        int64_t allocatedOffset = it->offset;
        assert(allocatedOffset % 256 == 0);
        if (it->size == sizeBytes) {
          blocks.erase(it);
        } else {
          it->offset += sizeBytes;
          it->size -= sizeBytes;
        }
        return allocatedOffset;
      }
    }
    llvm_unreachable("Free list ran out of address space");
  }

  void free(int64_t offset, int64_t sizeBytes) {
    assert(offset % 256 == 0);
    assert(sizeBytes % 256 == 0);
    FreeBlock newBlock{offset, sizeBytes};
    auto it = std::lower_bound(blocks.begin(), blocks.end(), newBlock,
                               [](const FreeBlock& a, const FreeBlock& b) {
                                 return a.offset < b.offset;
                               });
    it = blocks.insert(it, newBlock);

    // Merge with next block if adjacent
    auto next = std::next(it);
    if (next != blocks.end() && it->offset + it->size == next->offset) {
      it->size += next->size;
      blocks.erase(next);
    }
    // Merge with prev block if adjacent
    if (it != blocks.begin()) {
      auto prev = std::prev(it);
      if (prev->offset + prev->size == it->offset) {
        prev->size += it->size;
        blocks.erase(it);
      }
    }
  }

 private:
  std::vector<FreeBlock> blocks;
};

struct AllocInterval {
  memref::AllocOp allocOp;
  int64_t sizeBytes;
  int defPoint;
  int lastUsePoint;
  int64_t localOffset = 0;
};

static LogicalResult computeArenaCallGraphTopoOrder(
    ModuleOp module, ArrayRef<func::FuncOp> allFuncs,
    llvm::DenseMap<Operation*, SmallVector<Operation*>>& callers,
    SmallVector<Operation*>& topoOrder) {
  llvm::DenseMap<Operation*, SmallVector<Operation*>> callees;
  llvm::DenseMap<Operation*, int> inDegree;
  for (func::FuncOp f : allFuncs) {
    callers[f];
    callees[f];
    inDegree[f] = 0;
  }

  for (func::FuncOp f : allFuncs) {
    f.walk([&](func::CallOp callOp) {
      if (auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
        if (!callee.isExternal() && !callee.empty()) {
          callees[f].push_back(callee);
          callers[callee].push_back(f);
          inDegree[callee]++;
        }
      }
    });
  }

  SmallVector<Operation*> worklist;
  for (func::FuncOp f : allFuncs) {
    if (inDegree[f] == 0) worklist.push_back(f);
  }

  while (!worklist.empty()) {
    Operation* curr = worklist.pop_back_val();
    topoOrder.push_back(curr);
    for (Operation* callee : callees[curr]) {
      if (--inDegree[callee] == 0) {
        worklist.push_back(callee);
      }
    }
  }

  if (topoOrder.size() < allFuncs.size()) {
    return module.emitError(
        "recursion is not supported in cheddar GPU arena allocation");
  }
  return success();
}

static LogicalResult convertHostResidencyAndAllocs(ModuleOp module) {
  MLIRContext* ctx = module.getContext();
  Location loc = module.getLoc();
  Type llvmPtrType = LLVM::LLVMPointerType::get(ctx);
  Type i32Type = IntegerType::get(ctx, 32);
  Type i1Type = IntegerType::get(ctx, 1);

  // 1. Gather all unique globals consumed by GPU
  llvm::MapVector<FlatSymbolRefAttr, memref::GlobalOp> uniqueGlobals;
  llvm::DenseMap<Operation*, llvm::MapVector<FlatSymbolRefAttr,
                                             SmallVector<memref::GetGlobalOp>>>
      funcGlobals;

  // Collect all non-external, non-empty functions in module
  SmallVector<func::FuncOp> allFuncs;
  for (Operation& op : module.getOps()) {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp || funcOp.isExternal() || funcOp.empty()) continue;
    if (funcOp.getName() == "cudaDeviceSynchronize" ||
        funcOp.getName() == "__cheddar_gpu_init")
      continue;
    allFuncs.push_back(funcOp);
  }

  for (func::FuncOp funcOp : allFuncs) {
    if (llvm::any_of(funcOp.getFunctionType().getResults(),
                     [](Type t) { return isa<MemRefType>(t); })) {
      return funcOp.emitError(
          "functions returning memref are not supported in cheddar GPU arena "
          "allocation");
    }
  }

  // 2. Gather globals and static/dynamic allocs for each function
  struct FuncAllocInfo {
    SmallVector<memref::AllocOp> staticAllocs;
    SmallVector<int64_t> staticOffsets;
    int64_t totalBytes = 0;
    SmallVector<memref::AllocOp> dynamicAllocs;
  };
  llvm::DenseMap<Operation*, FuncAllocInfo> funcAllocs;
  llvm::DenseMap<Operation*, int64_t> funcHwmMap;
  llvm::DenseMap<Operation*, SmallVector<AllocInterval>> funcIntervalsMap;

  for (func::FuncOp funcOp : allFuncs) {
    funcOp.walk([&](memref::GetGlobalOp getGlobal) {
      if (isBufferConsumedByGpu(getGlobal.getResult())) {
        FlatSymbolRefAttr nameAttr = getGlobal.getNameAttr();
        if (!uniqueGlobals.count(nameAttr)) {
          if (auto globalOp = module.lookupSymbol<memref::GlobalOp>(nameAttr)) {
            uniqueGlobals[nameAttr] = globalOp;
          }
        }
        funcGlobals[funcOp][nameAttr].push_back(getGlobal);
      }
    });

    int currentIndex = 0;
    llvm::DenseMap<Operation*, std::pair<int, int>> opRange;
    std::function<void(Operation*)> indexOp = [&](Operation* op) {
      int start = currentIndex++;
      for (Region& r : op->getRegions()) {
        for (Block& b : r) {
          for (Operation& nested : b) {
            indexOp(&nested);
          }
        }
      }
      int end = currentIndex++;
      opRange[op] = {start, end};
    };

    for (Operation& op : funcOp.getBody().front()) {
      indexOp(&op);
    }

    FuncAllocInfo allocInfo;
    SmallVector<AllocInterval> allocIntervals;

    WalkResult walkRes = funcOp.walk([&](memref::AllocOp allocOp)
                                         -> WalkResult {
      if (isBufferConsumedByGpu(allocOp.getResult())) {
        if (allocOp.getType().hasStaticShape()) {
          int64_t numElements = allocOp.getType().getNumElements();
          int64_t elemBytes =
              (allocOp.getType().getElementTypeBitWidth() + 7) / 8;
          if (numElements < 0 || elemBytes <= 0 ||
              (numElements > 0 &&
               numElements > std::numeric_limits<int64_t>::max() / elemBytes)) {
            allocOp.emitError("allocation size overflows int64_t");
            return WalkResult::interrupt();
          }
          int64_t rawBytes = numElements * elemBytes;
          if (rawBytes > std::numeric_limits<int64_t>::max() - 256) {
            allocOp.emitError("allocation size overflows int64_t");
            return WalkResult::interrupt();
          }
          int64_t sizeBytes = llvm::alignTo(rawBytes, 256);
          if (sizeBytes == 0) sizeBytes = 256;

          Operation* loopParent = allocOp->getParentOp();
          llvm::SmallPtrSet<Operation*, 4> enclosingLoops;
          while (loopParent && loopParent != funcOp) {
            if (isa<scf::ForOp, scf::WhileOp>(loopParent)) {
              enclosingLoops.insert(loopParent);
            }
            loopParent = loopParent->getParentOp();
          }

          AllocInterval interval;
          interval.allocOp = allocOp;
          interval.sizeBytes = sizeBytes;
          interval.defPoint = opRange[allocOp].first;
          interval.lastUsePoint = interval.defPoint;

          SmallVector<Value> aliasWorklist = {allocOp.getResult()};
          llvm::DenseSet<Value> visitedAliases = {allocOp.getResult()};
          bool escapesEnclosingLoop = false;

          while (!aliasWorklist.empty()) {
            Value curr = aliasWorklist.pop_back_val();

            if (auto blockArg = dyn_cast<BlockArgument>(curr)) {
              Operation* ownerOp = blockArg.getOwner()->getParentOp();
              if (ownerOp && ownerOp != funcOp) {
                Operation* topOp = ownerOp;
                while (topOp->getParentOp() && topOp->getParentOp() != funcOp) {
                  topOp = topOp->getParentOp();
                }
                auto it = opRange.find(topOp);
                if (it != opRange.end()) {
                  interval.lastUsePoint =
                      std::max(interval.lastUsePoint, it->second.second);
                }
              }
            }

            for (OpOperand& use : curr.getUses()) {
              Operation* user = use.getOwner();

              if (isa<memref::CastOp, memref::SubViewOp, memref::ViewOp,
                      memref::ReinterpretCastOp, memref::CollapseShapeOp,
                      memref::ExpandShapeOp>(user)) {
                for (Value res : user->getResults()) {
                  if (visitedAliases.insert(res).second) {
                    aliasWorklist.push_back(res);
                  }
                }
              } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
                for (auto [initArg, iterArg] :
                     llvm::zip(forOp.getInits(), forOp.getRegionIterArgs())) {
                  if (initArg == curr &&
                      visitedAliases.insert(iterArg).second) {
                    aliasWorklist.push_back(iterArg);
                  }
                }
                for (auto [res, initArg] :
                     llvm::zip(forOp.getResults(), forOp.getInits())) {
                  if (initArg == curr && visitedAliases.insert(res).second) {
                    aliasWorklist.push_back(res);
                  }
                }
              } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
                Operation* parentOp = yieldOp->getParentOp();
                if (enclosingLoops.contains(parentOp)) {
                  escapesEnclosingLoop = true;
                }
                if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
                  unsigned opIdx = use.getOperandNumber();
                  if (opIdx < ifOp.getNumResults()) {
                    Value ifRes = ifOp.getResult(opIdx);
                    if (visitedAliases.insert(ifRes).second) {
                      aliasWorklist.push_back(ifRes);
                    }
                  }
                } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
                  unsigned opIdx = use.getOperandNumber();
                  if (opIdx < forOp.getNumResults()) {
                    Value forRes = forOp.getResult(opIdx);
                    if (visitedAliases.insert(forRes).second) {
                      aliasWorklist.push_back(forRes);
                    }
                  }
                }
              } else if (auto condOp = dyn_cast<scf::ConditionOp>(user)) {
                Operation* parentOp = condOp->getParentOp();
                if (enclosingLoops.contains(parentOp)) {
                  escapesEnclosingLoop = true;
                }
              }

              Operation* topOp = user;
              while (topOp->getParentOp() && topOp->getParentOp() != funcOp) {
                topOp = topOp->getParentOp();
              }
              auto it = opRange.find(topOp);
              if (it != opRange.end()) {
                interval.lastUsePoint =
                    std::max(interval.lastUsePoint, it->second.second);
              }
            }
          }

          if (escapesEnclosingLoop) {
            allocInfo.dynamicAllocs.push_back(allocOp);
            return WalkResult::advance();
          }

          allocIntervals.push_back(interval);
        } else {
          allocInfo.dynamicAllocs.push_back(allocOp);
        }
      }
      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted()) return failure();

    llvm::sort(allocIntervals,
               [](const AllocInterval& a, const AllocInterval& b) {
                 return a.defPoint < b.defPoint;
               });

    struct ActiveBuffer {
      int64_t offset;
      int64_t sizeBytes;
      int lastUsePoint;
    };
    SmallVector<ActiveBuffer> activeBuffers;
    FirstFitFreeList freeList;
    int64_t funcHwm = 0;

    for (AllocInterval& entry : allocIntervals) {
      SmallVector<ActiveBuffer> remainingActive;
      for (const ActiveBuffer& act : activeBuffers) {
        if (act.lastUsePoint < entry.defPoint) {
          freeList.free(act.offset, act.sizeBytes);
        } else {
          remainingActive.push_back(act);
        }
      }
      activeBuffers = std::move(remainingActive);

      int64_t localOffset = freeList.allocate(entry.sizeBytes);
      entry.localOffset = localOffset;
      funcHwm = std::max(funcHwm, localOffset + entry.sizeBytes);
      activeBuffers.push_back(
          {localOffset, entry.sizeBytes, entry.lastUsePoint});
    }

    funcHwmMap[funcOp] = funcHwm;
    funcIntervalsMap[funcOp] = std::move(allocIntervals);
    funcAllocs[funcOp] = std::move(allocInfo);
  }

  // Build Call Graph DAG
  llvm::DenseMap<Operation*, SmallVector<Operation*>> callers;
  SmallVector<Operation*> topoOrder;
  if (failed(computeArenaCallGraphTopoOrder(module, allFuncs, callers,
                                            topoOrder))) {
    return failure();
  }

  llvm::DenseMap<Operation*, int64_t> baseMap;
  int64_t globalArenaBytes = 0;

  for (Operation* op : topoOrder) {
    auto funcOp = cast<func::FuncOp>(op);
    int64_t base = 0;
    if (!callers[funcOp].empty()) {
      for (Operation* caller : callers[funcOp]) {
        int64_t callerEnd = baseMap[caller] + funcHwmMap[caller];
        base = std::max<int64_t>(
            base, static_cast<int64_t>(llvm::alignTo(callerEnd, 256)));
      }
    }
    baseMap[funcOp] = base;
    globalArenaBytes = std::max(globalArenaBytes, base + funcHwmMap[funcOp]);
  }

  for (func::FuncOp funcOp : allFuncs) {
    FuncAllocInfo& aInfo = funcAllocs[funcOp];
    int64_t base = baseMap[funcOp];
    int64_t hwm = funcHwmMap[funcOp];
    aInfo.totalBytes = hwm;

    for (const AllocInterval& interval : funcIntervalsMap[funcOp]) {
      aInfo.staticAllocs.push_back(interval.allocOp);
      aInfo.staticOffsets.push_back(base + interval.localOffset);
    }

    LLVM_DEBUG({
      int64_t bumpTotal = 0;
      for (const AllocInterval& interval : funcIntervalsMap[funcOp]) {
        bumpTotal += interval.sizeBytes;
      }
      llvm::dbgs() << "[cheddar-gpu-arena] " << funcOp.getName()
                   << ": base=" << base << ", hwm=" << hwm
                   << ", bump=" << bumpTotal << ", total=" << globalArenaBytes
                   << "\n";
    });
  }

  int64_t maxArenaBytes = globalArenaBytes;

  bool needInit = (!uniqueGlobals.empty() || maxArenaBytes > 0);

  OpBuilder modB(&module.getBodyRegion().front(),
                 module.getBodyRegion().front().begin());

  LLVM::GlobalOp initFlagGlobal;
  LLVM::GlobalOp arenaPtrGlobal;
  llvm::DenseMap<FlatSymbolRefAttr, LLVM::GlobalOp> devPtrGlobals;
  func::FuncOp initFunc;

  if (needInit) {
    // 1. Init flag: __cheddar_gpu_init_flag
    std::string initFlagName = "__cheddar_gpu_init_flag";
    if (!(initFlagGlobal = module.lookupSymbol<LLVM::GlobalOp>(initFlagName))) {
      initFlagGlobal = LLVM::GlobalOp::create(
          modB, loc, i1Type, /*isConstant=*/false, LLVM::Linkage::Internal,
          initFlagName, modB.getBoolAttr(false));
    }

    // 2. Global scratch arena pointer if maxArenaBytes > 0
    if (maxArenaBytes > 0) {
      module->setAttr("cheddar_gpu.arena_size",
                      modB.getI64IntegerAttr(maxArenaBytes));
      std::string arenaPtrName = "__cheddar_arena_ptr";
      if (!(arenaPtrGlobal =
                module.lookupSymbol<LLVM::GlobalOp>(arenaPtrName))) {
        arenaPtrGlobal = LLVM::GlobalOp::create(
            modB, loc, llvmPtrType, /*isConstant=*/false,
            LLVM::Linkage::Internal, arenaPtrName, Attribute());
        Block* initBlock = new Block();
        arenaPtrGlobal.getInitializerRegion().push_back(initBlock);
        OpBuilder initB(initBlock, initBlock->begin());
        Value nullPtr = LLVM::ZeroOp::create(initB, loc, llvmPtrType);
        LLVM::ReturnOp::create(initB, loc, nullPtr);
      }
    }

    // 3. Global device pointers for each table
    for (auto& [nameAttr, hostGlobalOp] : uniqueGlobals) {
      std::string devPtrName =
          ("__cheddar_dev_ptr_" + nameAttr.getValue()).str();
      LLVM::GlobalOp devPtrGlobal =
          module.lookupSymbol<LLVM::GlobalOp>(devPtrName);
      if (!devPtrGlobal) {
        devPtrGlobal = LLVM::GlobalOp::create(
            modB, loc, llvmPtrType, /*isConstant=*/false,
            LLVM::Linkage::Internal, devPtrName, Attribute());
        Block* initBlock = new Block();
        devPtrGlobal.getInitializerRegion().push_back(initBlock);
        OpBuilder initB(initBlock, initBlock->begin());
        Value nullPtr = LLVM::ZeroOp::create(initB, loc, llvmPtrType);
        LLVM::ReturnOp::create(initB, loc, nullPtr);
      }
      devPtrGlobals[nameAttr] = devPtrGlobal;
    }

    // 4. Create __cheddar_gpu_init
    std::string initFuncName = "__cheddar_gpu_init";
    initFunc = module.lookupSymbol<func::FuncOp>(initFuncName);
    if (!initFunc) {
      FunctionType initFuncType = FunctionType::get(ctx, {}, {});
      initFunc = func::FuncOp::create(modB, loc, initFuncName, initFuncType);
      initFunc->setAttr("sym_visibility", modB.getStringAttr("private"));
      initFunc->setAttr(
          "llvm.linkage",
          LLVM::LinkageAttr::get(ctx, LLVM::linkage::Linkage::Internal));
      initFunc->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
      Block* entryBlock = initFunc.addEntryBlock();
      OpBuilder b(entryBlock, entryBlock->begin());

      Value flagAddr = LLVM::AddressOfOp::create(b, loc, initFlagGlobal);
      Value flag = LLVM::LoadOp::create(b, loc, i1Type, flagAddr);
      Value c1I1 =
          LLVM::ConstantOp::create(b, loc, i1Type, b.getBoolAttr(true));
      Value isInit =
          LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq, flag, c1I1);
      Value isNotInit = arith::XOrIOp::create(b, loc, isInit, c1I1);

      auto ifOp =
          scf::IfOp::create(b, loc, isNotInit, /*withElseRegion=*/false);
      OpBuilder thenB(&ifOp.getThenRegion().front(),
                      ifOp.getThenRegion().front().begin());

      LLVM::StoreOp::create(thenB, loc, c1I1, flagAddr);

      // Allocate persistent scratch arena if maxArenaBytes > 0
      if (maxArenaBytes > 0) {
        auto arenaMemRefType = MemRefType::get({maxArenaBytes}, b.getI8Type());
        auto gpuAlloc = gpu::AllocOp::create(
            thenB, loc, arenaMemRefType, /*asyncDependencies=*/ValueRange{},
            /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{});
        Value rawArenaPtr = extractRawPointerFromDeviceMemRef(
            thenB, loc, gpuAlloc.getMemref(), arenaMemRefType);
        Value arenaAddr = LLVM::AddressOfOp::create(thenB, loc, arenaPtrGlobal);
        LLVM::StoreOp::create(thenB, loc, rawArenaPtr, arenaAddr);
      }

      if (!uniqueGlobals.empty()) {
        auto asyncTokenType = gpu::AsyncTokenType::get(ctx);
        Value currentToken =
            gpu::WaitOp::create(thenB, loc, asyncTokenType, ValueRange{})
                .getAsyncToken();
        for (auto& [nameAttr, hostGlobalOp] : uniqueGlobals) {
          auto memrefType = cast<MemRefType>(hostGlobalOp.getType());
          auto gpuAlloc = gpu::AllocOp::create(
              thenB, loc, memrefType, /*asyncDependencies=*/ValueRange{},
              /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{});
          auto hostGetGlobal =
              memref::GetGlobalOp::create(thenB, loc, memrefType, nameAttr);
          auto memcpyOp = gpu::MemcpyOp::create(
              thenB, loc, asyncTokenType, ValueRange{currentToken},
              gpuAlloc.getMemref(), hostGetGlobal.getResult());
          currentToken = memcpyOp.getAsyncToken();

          Value rawPtr = extractRawPointerFromDeviceMemRef(
              thenB, loc, gpuAlloc.getMemref(), memrefType);
          Value tableAddr =
              LLVM::AddressOfOp::create(thenB, loc, devPtrGlobals[nameAttr]);
          LLVM::StoreOp::create(thenB, loc, rawPtr, tableAddr);
        }
        // Host-BLOCKING wait on the async token chain with no result
        gpu::WaitOp::create(thenB, loc, Type(), ValueRange{currentToken});
      }

      std::string syncFuncName = "cudaDeviceSynchronize";
      func::FuncOp syncFunc = module.lookupSymbol<func::FuncOp>(syncFuncName);
      if (!syncFunc) {
        FunctionType syncFuncType = FunctionType::get(ctx, {}, {i32Type});
        syncFunc = func::FuncOp::create(modB, loc, syncFuncName, syncFuncType);
        syncFunc.setPrivate();
      }
      func::CallOp::create(thenB, loc, syncFunc, ValueRange{});

      func::ReturnOp::create(b, loc);
    }
  }

  // 5. In each funcOp, hook up lazy init, persistent arena views, and device
  // table pointers
  for (Operation& op : llvm::make_early_inc_range(module.getOps())) {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp) continue;
    if (initFunc && funcOp == initFunc) continue;
    if (funcOp.getSymName() == "cudaDeviceSynchronize") continue;
    if (funcOp.isExternal() || funcOp.empty()) continue;

    auto allocIt = funcAllocs.find(funcOp);
    bool hasStaticAllocs =
        (allocIt != funcAllocs.end() && !allocIt->second.staticAllocs.empty());
    auto globalIt = funcGlobals.find(funcOp);
    bool hasGlobals =
        (globalIt != funcGlobals.end() && !globalIt->second.empty());

    Block& entryBlock = funcOp.front();
    OpBuilder b(&entryBlock, entryBlock.begin());

    if (initFunc) {
      // Call __cheddar_gpu_init()
      func::CallOp::create(b, loc, initFunc, ValueRange{});
    }

    // Force all kernel launches to execute on the legacy default stream (stream
    // 0) by providing a null stream pointer as the asyncObject. This bypasses
    // SelectObjectAttr's default mgpuStreamCreate/Destroy path (which creates
    // transient CU_STREAM_NON_BLOCKING streams that do not serialize with
    // stream 0).
    Value nullStream = LLVM::ZeroOp::create(b, loc, llvmPtrType);
    funcOp.walk([&](gpu::LaunchFuncOp launchOp) {
      if (!launchOp.getAsyncObject()) {
        launchOp.getAsyncObjectMutable().assign(nullStream);
      }
    });

    if (hasStaticAllocs) {
      FuncAllocInfo& aInfo = allocIt->second;
      Value arenaAddr = LLVM::AddressOfOp::create(b, loc, arenaPtrGlobal);
      Value rawArenaPtr = LLVM::LoadOp::create(b, loc, llvmPtrType, arenaAddr);
      auto funcArenaType = MemRefType::get({maxArenaBytes}, b.getI8Type());
      Value arena = reconstructDeviceMemRef(b, loc, funcArenaType, rawArenaPtr);

      for (size_t i = 0; i < aInfo.staticAllocs.size(); ++i) {
        memref::AllocOp allocOp = aInfo.staticAllocs[i];
        OpBuilder allocB(allocOp);
        Value offsetVal = arith::ConstantIndexOp::create(
            allocB, allocOp.getLoc(), aInfo.staticOffsets[i]);
        Value view =
            memref::ViewOp::create(allocB, allocOp.getLoc(), allocOp.getType(),
                                   arena, offsetVal, ValueRange{});
        allocOp.replaceAllUsesWith(view);
        allocOp.erase();
      }
    }

    if (hasGlobals) {
      SmallVector<memref::GetGlobalOp> opsToErase;
      for (auto& [nameAttr, getGlobalOps] : globalIt->second) {
        auto devPtrIt = devPtrGlobals.find(nameAttr);
        if (devPtrIt == devPtrGlobals.end() || !devPtrIt->second) continue;
        Value tableAddr = LLVM::AddressOfOp::create(b, loc, devPtrIt->second);
        Value rawPtr = LLVM::LoadOp::create(b, loc, llvmPtrType, tableAddr);
        auto memrefType = cast<MemRefType>(getGlobalOps.front().getType());
        Value devMemRef = reconstructDeviceMemRef(b, loc, memrefType, rawPtr);

        for (memref::GetGlobalOp getGlobal : getGlobalOps) {
          getGlobal.replaceAllUsesWith(devMemRef);
          opsToErase.push_back(getGlobal);
        }
      }
      for (memref::GetGlobalOp getGlobal : opsToErase) {
        getGlobal.erase();
      }
    }

    // Dynamic allocs if any
    if (allocIt != funcAllocs.end() && !allocIt->second.dynamicAllocs.empty()) {
      for (memref::AllocOp allocOp : allocIt->second.dynamicAllocs) {
        OpBuilder b(allocOp);
        auto gpuAlloc = gpu::AllocOp::create(
            b, allocOp.getLoc(), allocOp.getType(),
            /*asyncDependencies=*/ValueRange{}, allocOp.getDynamicSizes(),
            allocOp.getSymbolOperands());
        allocOp.replaceAllUsesWith(gpuAlloc.getMemref());

        bool hasDealloc = false;
        for (Operation* user : gpuAlloc.getMemref().getUsers()) {
          if (isa<memref::DeallocOp, gpu::DeallocOp>(user)) {
            hasDealloc = true;
            break;
          }
        }
        if (!hasDealloc) {
          Block* block = allocOp->getBlock();
          Operation* terminator = block->getTerminator();
          OpBuilder deallocBuilder(terminator ? terminator : &block->back());
          gpu::DeallocOp::create(deallocBuilder, allocOp.getLoc(), ValueRange{},
                                 gpuAlloc.getMemref());
        }
        allocOp.erase();
      }
    }
  }

  // 6. Erase any dead memref.dealloc
  SmallVector<memref::DeallocOp> deadDeallocs;
  module.walk(
      [&](memref::DeallocOp deallocOp) { deadDeallocs.push_back(deallocOp); });
  for (auto deallocOp : deadDeallocs) {
    deallocOp.erase();
  }
  return success();
}

// -----------------------------------------------------------------------------
// Pre-lowering Copy Elimination (Alias-Safe Producer Forwarding)
// -----------------------------------------------------------------------------
//
// Eliminates redundant `cheddar_gpu.copy(dst, src)` where:
// 1. `src` is defined by a local `memref.alloc` in the same block as `copyOp`.
// 2. `copyOp` is a full-buffer copy (`!in_limb_range` && `!out_limb_range` &&
//    `dst.getType() == src.getType()`).
// 3. `src` is used solely by a single producer op `P` (which writes to `src`)
//    and `copyOp` (which reads from `src`), plus optional dead
//    `memref.dealloc`.
// 4. `P` dominates `copyOp` in the same block and does not read or write `dst`
//    (or any alias of `dst`).
// 5. No operation between `P` and `copyOp` reads or writes `dst` or any alias
//    of `dst` (verified via root base-object tracking through `cast`,
//    `subview`, `view`, and `reinterpret_cast`).
// 6. There are no function calls (`func.call` or `CallOpInterface`) between `P`
//    and `copyOp`.
//
// Under these conditions, `P` writing `dst` directly instead of `src` is
// provably alias-safe, bit-exact, and preserves program semantics while saving
// an allocation and a kernel launch.

static Value getUnderlyingBaseObject(Value v) {
  while (true) {
    if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
      v = castOp.getSource();
    } else if (auto subviewOp = v.getDefiningOp<memref::SubViewOp>()) {
      v = subviewOp.getSource();
    } else if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
      v = viewOp.getSource();
    } else if (auto rcastOp = v.getDefiningOp<memref::ReinterpretCastOp>()) {
      v = rcastOp.getSource();
    } else {
      break;
    }
  }
  return v;
}

static bool tryForwardCopy(CopyOp copyOp) {
  if (copyOp.getInLimbRange().has_value() ||
      copyOp.getOutLimbRange().has_value()) {
    return false;
  }

  Value dst = copyOp.getOut();
  Value src = copyOp.getIn();
  if (dst.getType() != src.getType()) return false;

  auto allocOp = src.getDefiningOp<memref::AllocOp>();
  if (!allocOp || allocOp->getBlock() != copyOp->getBlock()) return false;

  Operation* producer = nullptr;
  SmallVector<memref::DeallocOp> deallocs;
  for (Operation* user : src.getUsers()) {
    if (user == copyOp.getOperation()) {
      continue;
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocs.push_back(dealloc);
      continue;
    }
    if (producer != nullptr) {
      // More than one non-copy/dealloc user; cannot forward.
      return false;
    }
    producer = user;
  }

  if (!producer || producer->getBlock() != copyOp->getBlock() ||
      !producer->isBeforeInBlock(copyOp.getOperation())) {
    return false;
  }

  // Find the unique operand of producer that equals src.
  std::optional<unsigned> srcOperandIdx;
  for (auto [idx, opnd] : llvm::enumerate(producer->getOperands())) {
    if (opnd == src) {
      if (srcOperandIdx.has_value()) {
        // src appears multiple times in producer.
        return false;
      }
      srcOperandIdx = idx;
    }
  }
  if (!srcOperandIdx.has_value()) return false;

  // Verify that producer writes to src and does not read it.
  if (producer->getDialect()->getNamespace() != "cheddar_gpu") {
    return false;
  }
  // For CheddarGPU ops, outputs are the leading operands:
  // 1-output ops: index 0
  // 2-output ops: index 0 or 1
  // 3-output ops (TensorOp, TensorSquareOp): index 0, 1, or 2
  bool isOutput = false;
  if (isa<Add2Op, Sub2Op, Neg2Op, CPAccum2Op, CPAccumAdd2Op, HRotOp, Permute2Op,
          PermuteAccum2Op>(producer)) {
    isOutput = (*srcOperandIdx <= 1);
  } else if (isa<TensorOp, TensorSquareOp>(producer)) {
    isOutput = (*srcOperandIdx <= 2);
  } else {
    isOutput = (*srcOperandIdx == 0);
  }
  if (!isOutput) return false;

  // Ensure producer does not already use dst or an alias of dst.
  Value rootDst = getUnderlyingBaseObject(dst);
  for (Value opnd : producer->getOperands()) {
    if (isa<MemRefType>(opnd.getType()) &&
        getUnderlyingBaseObject(opnd) == rootDst) {
      return false;
    }
  }

  // Check all operations between producer and copyOp.
  for (Operation* curr = producer->getNextNode(); curr != copyOp.getOperation();
       curr = curr->getNextNode()) {
    bool hasCall = false;
    curr->walk([&](Operation* nested) {
      if (isa<func::CallOp>(nested) ||
          nested->getName().getStringRef() == "func.call") {
        hasCall = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasCall) return false;

    for (Value opnd : curr->getOperands()) {
      if (isa<MemRefType>(opnd.getType()) &&
          getUnderlyingBaseObject(opnd) == rootDst) {
        return false;
      }
    }
    for (Value res : curr->getResults()) {
      if (isa<MemRefType>(res.getType()) &&
          getUnderlyingBaseObject(res) == rootDst) {
        return false;
      }
    }
  }

  // All alias-safety checks satisfied: rewrite producer and erase copy + alloc.
  producer->setOperand(*srcOperandIdx, dst);
  copyOp.erase();
  for (auto dealloc : deallocs) {
    dealloc.erase();
  }
  allocOp.erase();
  return true;
}

static void forwardAllocCopies(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    module.walk([&](CopyOp copyOp) {
      if (tryForwardCopy(copyOp)) {
        changed = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
}

struct CheddarGPUToGPU : public impl::CheddarGPUToGPUBase<CheddarGPUToGPU> {
  using impl::CheddarGPUToGPUBase<CheddarGPUToGPU>::CheddarGPUToGPUBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ModuleOp module = getOperation();

    forwardAllocCopies(module);

    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (paramsAttr) {
      gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);
      getOrCreateDevicePrimeGlobals(gpuMod, paramsAttr, paramsAttr.getWord());
      unsigned w = paramsAttr.getWord();
      int64_t degree = 1LL << paramsAttr.getLogN();
      if (w == 32) {
        getOrCreateModuleTwiddleGlobals<32>(module, paramsAttr, degree);
      } else if (w == 64) {
        getOrCreateModuleTwiddleGlobals<64>(module, paramsAttr, degree);
      }
    }

    RewritePatternSet patterns(context);
    patterns.add<ConvertCopyOp, ConvertAddOp, ConvertSubOp, ConvertNegOp,
                 ConvertAdd2Op, ConvertSub2Op, ConvertNeg2Op, ConvertAddConstOp,
                 ConvertSubConstOp, ConvertSubOppositeConstOp, ConvertMulOp,
                 ConvertMulConstOp, ConvertCPAccumOp, ConvertCPAccumAddOp,
                 ConvertCPAccum2Op, ConvertCPAccumAdd2Op, ConvertTensorOp,
                 ConvertTensorSquareOp, ConvertMulImaginaryUnitOp,
                 ConvertPermuteOp, ConvertPermute2Op, ConvertPermuteAccumOp,
                 ConvertPermuteAccum2Op, ConvertZeroPadOp, ConvertModUpToMaxOp>(
        context);
    if (nttImpl == "fused") {
      populateNTTFusedToGPUPatterns(patterns);
    } else {
      populateNTTToGPUPatterns(patterns);
    }
    populateBConvToGPUPatterns(patterns);
    populateModDownEpilogueToGPUPatterns(patterns);
    populateHoistToGPUPatterns(patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           mod_arith::ModArithDialect>();
    target.addIllegalOp<
        CopyOp, AddOp, SubOp, NegOp, Add2Op, Sub2Op, Neg2Op, AddConstOp,
        SubConstOp, SubOppositeConstOp, MulOp, MulConstOp, CPAccumOp,
        CPAccumAddOp, CPAccum2Op, CPAccumAdd2Op, TensorOp, TensorSquareOp,
        MulImaginaryUnitOp, PermuteOp, Permute2Op, PermuteAccumOp,
        PermuteAccum2Op, ZeroPadOp, ModUpToMaxOp>();
    configureNTTGPUConversionTarget(target);
    configureBConvGPUConversionTarget(target);
    configureModDownEpilogueGPUConversionTarget(target);
    configureHoistGPUConversionTarget(target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    cloneGetGlobalsInsideLaunches(module);
    if (failed(convertHostResidencyAndAllocs(module))) {
      signalPassFailure();
      return;
    }
    module->removeAttr("cheddar_gpu.params");
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCheddarGPUToGPUPass() {
  return std::make_unique<CheddarGPUToGPU>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
