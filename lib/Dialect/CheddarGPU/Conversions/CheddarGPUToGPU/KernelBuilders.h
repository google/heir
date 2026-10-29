#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_KERNELBUILDERS_H_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_KERNELBUILDERS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/DLTI/DLTI.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

/// Module-level parameter struct extracted from module attribute.
struct ModuleParams {
  ParamsAttr params;
  uint32_t word;
  uint32_t logN;
  int64_t degree;
};

inline std::optional<ModuleParams> getModuleParams(Operation* op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return std::nullopt;
  auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
  if (!params) return std::nullopt;
  return ModuleParams{params, params.getWord(), params.getLogN(),
                      1LL << params.getLogN()};
}

template <unsigned W>
inline cheddar_ref::Params<W> paramsFromAttr(ParamsAttr attr) {
  cheddar_ref::Params<W> p;
  p.word_size = W;
  p.log_n = attr.getLogN();
  p.degree = 1ULL << p.log_n;
  p.alpha = attr.getAlpha();
  p.dnum = attr.getDnum();
  for (int64_t v : attr.getMain().asArrayRef()) {
    p.main_primes.push_back(
        static_cast<typename cheddar_ref::Params<W>::Word>(v));
  }
  for (int64_t v : attr.getAux().asArrayRef()) {
    p.aux_primes.push_back(
        static_cast<typename cheddar_ref::Params<W>::Word>(v));
  }
  for (int64_t v : attr.getTer().asArrayRef()) {
    p.ter_primes.push_back(
        static_cast<typename cheddar_ref::Params<W>::Word>(v));
  }
  size_t num_ter = p.ter_primes.size();
  for (size_t i = 0; i < num_ter; ++i) {
    p.all_primes.push_back(p.ter_primes[num_ter - 1 - i]);
  }
  for (auto prime : p.main_primes) p.all_primes.push_back(prime);
  for (auto prime : p.aux_primes) p.all_primes.push_back(prime);
  return p;
}

template <unsigned W>
inline const cheddar_ref::Tables<W>& getCachedTables(ParamsAttr attr) {
  static llvm::DenseMap<Attribute, std::unique_ptr<cheddar_ref::Tables<W>>>
      cache;
  auto it = cache.find(attr);
  if (it != cache.end()) {
    return *it->second;
  }
  auto params = paramsFromAttr<W>(attr);
  size_t numPrimes = params.all_primes.size();
  auto tables = std::make_unique<cheddar_ref::Tables<W>>(
      cheddar_ref::Tables<W>::Generate(params, numPrimes));
  auto* ptr = tables.get();
  cache[attr] = std::move(tables);
  return *ptr;
}

inline std::pair<int64_t, int64_t> getLimbRange(
    std::optional<ArrayRef<int64_t>> range, int64_t totalLimbs) {
  if (range && range->size() == 2) {
    return {(*range)[0], (*range)[1]};
  }
  return {0, totalLimbs};
}

/// Emits parallel 5-step bit reversal for 32-bit integer, reversed over
/// `numBits`.
inline Value buildBitReverse(ImplicitLocOpBuilder& b, Value val,
                             uint32_t numBits) {
  Type i32Ty = IntegerType::get(b.getContext(), 32);
  Value v = val;

  // Step 1: swap adjacent bits
  Value m1 = arith::ConstantIntOp::create(b, i32Ty, 0x55555555);
  Value s1_r =
      arith::ShRUIOp::create(b, v, arith::ConstantIntOp::create(b, i32Ty, 1));
  Value a1_r = arith::AndIOp::create(b, s1_r, m1);
  Value a1_l = arith::AndIOp::create(b, v, m1);
  Value s1_l =
      arith::ShLIOp::create(b, a1_l, arith::ConstantIntOp::create(b, i32Ty, 1));
  v = arith::OrIOp::create(b, a1_r, s1_l);

  // Step 2: swap 2-bit fields
  Value m2 = arith::ConstantIntOp::create(b, i32Ty, 0x33333333);
  Value s2_r =
      arith::ShRUIOp::create(b, v, arith::ConstantIntOp::create(b, i32Ty, 2));
  Value a2_r = arith::AndIOp::create(b, s2_r, m2);
  Value a2_l = arith::AndIOp::create(b, v, m2);
  Value s2_l =
      arith::ShLIOp::create(b, a2_l, arith::ConstantIntOp::create(b, i32Ty, 2));
  v = arith::OrIOp::create(b, a2_r, s2_l);

  // Step 3: swap 4-bit nibbles
  Value m4 = arith::ConstantIntOp::create(b, i32Ty, 0x0F0F0F0F);
  Value s4_r =
      arith::ShRUIOp::create(b, v, arith::ConstantIntOp::create(b, i32Ty, 4));
  Value a4_r = arith::AndIOp::create(b, s4_r, m4);
  Value a4_l = arith::AndIOp::create(b, v, m4);
  Value s4_l =
      arith::ShLIOp::create(b, a4_l, arith::ConstantIntOp::create(b, i32Ty, 4));
  v = arith::OrIOp::create(b, a4_r, s4_l);

  // Step 4: swap bytes
  Value m8 = arith::ConstantIntOp::create(b, i32Ty, 0x00FF00FF);
  Value s8_r =
      arith::ShRUIOp::create(b, v, arith::ConstantIntOp::create(b, i32Ty, 8));
  Value a8_r = arith::AndIOp::create(b, s8_r, m8);
  Value a8_l = arith::AndIOp::create(b, v, m8);
  Value s8_l =
      arith::ShLIOp::create(b, a8_l, arith::ConstantIntOp::create(b, i32Ty, 8));
  v = arith::OrIOp::create(b, a8_r, s8_l);

  // Step 5: swap 16-bit halves
  Value s16_r =
      arith::ShRUIOp::create(b, v, arith::ConstantIntOp::create(b, i32Ty, 16));
  Value s16_l =
      arith::ShLIOp::create(b, v, arith::ConstantIntOp::create(b, i32Ty, 16));
  v = arith::OrIOp::create(b, s16_r, s16_l);

  // Shift right to align to numBits
  if (32 > numBits) {
    v = arith::ShRUIOp::create(
        b, v, arith::ConstantIntOp::create(b, i32Ty, 32 - numBits));
  }
  return v;
}

/// Emits canonical modular addition: (a + b >= q) ? (a + b - q) : (a + b)
inline Value buildModAdd(ImplicitLocOpBuilder& b, Value a, Value bVal,
                         uint64_t q) {
  Type intType = a.getType();
  Value qVal = arith::ConstantIntOp::create(b, intType, q);
  Value sum = arith::AddIOp::create(b, a, bVal);
  Value isGe = arith::CmpIOp::create(b, arith::CmpIPredicate::uge, sum, qVal);
  Value sumSubQ = arith::SubIOp::create(b, sum, qVal);
  return arith::SelectOp::create(b, isGe, sumSubQ, sum);
}

/// Emits canonical modular subtraction: (a < b) ? (a - b + q) : (a - b)
inline Value buildModSub(ImplicitLocOpBuilder& b, Value a, Value bVal,
                         uint64_t q) {
  Type intType = a.getType();
  Value qVal = arith::ConstantIntOp::create(b, intType, q);
  Value isLt = arith::CmpIOp::create(b, arith::CmpIPredicate::ult, a, bVal);
  Value diff = arith::SubIOp::create(b, a, bVal);
  Value diffAddQ = arith::AddIOp::create(b, diff, qVal);
  return arith::SelectOp::create(b, isLt, diffAddQ, diff);
}

/// Emits canonical modular negation: (a > 0) ? (q - a) : 0
inline Value buildModNeg(ImplicitLocOpBuilder& b, Value a, uint64_t q) {
  Type intType = a.getType();
  Value zero = arith::ConstantIntOp::create(b, intType, 0);
  Value qVal = arith::ConstantIntOp::create(b, intType, q);
  Value isGtZero = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, a, zero);
  Value qSubA = arith::SubIOp::create(b, qVal, a);
  return arith::SelectOp::create(b, isGtZero, qSubA, zero);
}

/// Emits Montgomery multiplication using mod_arith ops:
/// encapsulate -> mont_mul -> lift standard
inline Value buildMontMul(
    ImplicitLocOpBuilder& b, Value a, Value bVal,
    mod_arith::ModArithType modType,
    mod_arith::LiftType liftType = mod_arith::LiftType::STANDARD) {
  Type elemType = a.getType();
  Value aMod = mod_arith::EncapsulateOp::create(b, modType, a);
  Value bMod = mod_arith::EncapsulateOp::create(b, modType, bVal);
  Value prodMod = mod_arith::MontMulOp::create(b, modType, aMod, bMod);
  return mod_arith::LiftOp::create(b, elemType, prodMod, liftType);
}

/// Runtime modular addition: (a + b >= q) ? (a + b - q) : (a + b)
inline Value buildModAddRuntime(ImplicitLocOpBuilder& b, Value a, Value bVal,
                                Value q) {
  Value sum = arith::AddIOp::create(b, a, bVal);
  Value isGe = arith::CmpIOp::create(b, arith::CmpIPredicate::uge, sum, q);
  Value sumSubQ = arith::SubIOp::create(b, sum, q);
  return arith::SelectOp::create(b, isGe, sumSubQ, sum);
}

/// Runtime modular subtraction: (a < b) ? (a - b + q) : (a - b)
inline Value buildModSubRuntime(ImplicitLocOpBuilder& b, Value a, Value bVal,
                                Value q) {
  Value isLt = arith::CmpIOp::create(b, arith::CmpIPredicate::ult, a, bVal);
  Value diff = arith::SubIOp::create(b, a, bVal);
  Value diffAddQ = arith::AddIOp::create(b, diff, q);
  return arith::SelectOp::create(b, isLt, diffAddQ, diff);
}

/// Runtime modular negation: (a > 0) ? (q - a) : 0
inline Value buildModNegRuntime(ImplicitLocOpBuilder& b, Value a, Value q) {
  Type intType = a.getType();
  Value zero = arith::ConstantIntOp::create(b, intType, 0);
  Value isGtZero = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, a, zero);
  Value qSubA = arith::SubIOp::create(b, q, a);
  return arith::SelectOp::create(b, isGtZero, qSubA, zero);
}

/// Runtime Montgomery multiplication using Harvey reduction core:
/// lo = a * b, hi = mul_hi(a, b)
/// m = lo * q_inv
/// t = mul_hi(m, q)
/// r = hi - t
/// (r < 0) ? (r + q) : r
inline Value buildMontMulRuntime(ImplicitLocOpBuilder& b, Value a, Value bVal,
                                 Value q, Value qInv) {
  Type intType = a.getType();
  auto mulAB = arith::MulSIExtendedOp::create(b, a, bVal);
  Value lo = mulAB.getLow();
  Value hi = mulAB.getHigh();
  Value m = arith::MulIOp::create(b, lo, qInv);
  auto mulMQ = arith::MulSIExtendedOp::create(b, m, q);
  Value t = mulMQ.getHigh();
  Value r = arith::SubIOp::create(b, hi, t);
  Value zero = arith::ConstantIntOp::create(b, intType, 0);
  Value isNeg = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, r, zero);
  Value rPlusQ = arith::AddIOp::create(b, r, q);
  return arith::SelectOp::create(b, isNeg, rPlusQ, r);
}

/// Runtime Centered Lift: (r > (q >> 1)) ? (r - q) : r
inline Value buildCenteredRuntime(ImplicitLocOpBuilder& b, Value r, Value q) {
  Type intType = r.getType();
  Value one = arith::ConstantIntOp::create(b, intType, 1);
  Value halfQ = arith::ShRUIOp::create(b, q, one);
  Value isGt = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, r, halfQ);
  Value rMinusQ = arith::SubIOp::create(b, r, q);
  return arith::SelectOp::create(b, isGt, rMinusQ, r);
}

/// Runtime butterfly arithmetic for forward Cooley-Tukey NTT:
/// mult = v * w (mod q)
/// uNew = u + mult (mod q)
/// vNew = u - mult (mod q)
inline void butterflyNTTRuntime(ImplicitLocOpBuilder& b, Value& u, Value& v,
                                Value w, Value q, Value qInv) {
  Value mult = buildMontMulRuntime(b, v, w, q, qInv);
  Value uNew = buildModAddRuntime(b, u, mult, q);
  Value vNew = buildModSubRuntime(b, u, mult, q);
  u = uNew;
  v = vNew;
}

/// Runtime butterfly arithmetic for inverse Gentleman-Sande INTT:
/// uNew = u + v (mod q)
/// diff = u - v (mod q)
/// vNew = diff * w (mod q)
inline void butterflyINTTRuntime(ImplicitLocOpBuilder& b, Value& u, Value& v,
                                 Value w, Value q, Value qInv) {
  Value uNew = buildModAddRuntime(b, u, v, q);
  Value diff = buildModSubRuntime(b, u, v, q);
  Value vNew = buildMontMulRuntime(b, diff, w, q, qInv);
  u = uNew;
  v = vNew;
}

/// Runtime Montgomery reduction of wide integer (e.g. 2*W bits) mod q using
/// qInv: lo = trunc(wide, W) hi = trunc(wide >> W, W) m = lo * qInv t =
/// mul_hi(m, q) r = hi - t (r < 0) ? (r + q) : r
inline Value buildMontReduceRuntime(ImplicitLocOpBuilder& b, Value wide,
                                    Value q, Value qInv) {
  Type intType = q.getType();
  unsigned w = intType.getIntOrFloatBitWidth();
  Type wideType = wide.getType();
  Value lo = arith::TruncIOp::create(b, intType, wide);
  Value shiftVal = arith::ConstantIntOp::create(b, wideType, w);
  Value shifted = arith::ShRSIOp::create(b, wide, shiftVal);
  Value hi = arith::TruncIOp::create(b, intType, shifted);
  Value m = arith::MulIOp::create(b, lo, qInv);
  auto mulMQ = arith::MulSIExtendedOp::create(b, m, q);
  Value t = mulMQ.getHigh();
  Value r = arith::SubIOp::create(b, hi, t);
  Value zero = arith::ConstantIntOp::create(b, intType, 0);
  Value isNeg = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, r, zero);
  Value rPlusQ = arith::AddIOp::create(b, r, q);
  return arith::SelectOp::create(b, isNeg, rPlusQ, r);
}

/// Helper to cast the outermost dimension of a MemRef to dynamic.
inline Value castToDynamicFirstDim(OpBuilder& b, Location loc, Value val) {
  auto type = dyn_cast<MemRefType>(val.getType());
  if (!type || type.getRank() == 0 || type.isDynamicDim(0)) {
    return val;
  }
  SmallVector<int64_t> shape(type.getShape());
  shape[0] = ShapedType::kDynamic;
  auto dynamicType = MemRefType::get(shape, type.getElementType(),
                                     type.getLayout(), type.getMemorySpace());
  return memref::CastOp::create(b, loc, dynamicType, val);
}

/// Helper to cast a 1D MemRef to dynamic shape memref<?xiW>.
inline Value castToDynamic1D(OpBuilder& b, Location loc, Value val) {
  auto type = dyn_cast<MemRefType>(val.getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0)) {
    return val;
  }
  auto dynamicType =
      MemRefType::get({ShapedType::kDynamic}, type.getElementType(),
                      type.getLayout(), type.getMemorySpace());
  return memref::CastOp::create(b, loc, dynamicType, val);
}

/// Helper to cast a 2D MemRef to fully dynamic shape memref<?x?xiW>.
inline Value castToDynamic2D(OpBuilder& b, Location loc, Value val) {
  auto type = dyn_cast<MemRefType>(val.getType());
  if (!type || type.getRank() != 2) return val;
  if (type.isDynamicDim(0) && type.isDynamicDim(1)) return val;
  auto dynamicType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, type.getElementType(),
      type.getLayout(), type.getMemorySpace());
  return memref::CastOp::create(b, loc, dynamicType, val);
}

//===----------------------------------------------------------------------===//
// Kernel Library Infrastructure
//===----------------------------------------------------------------------===//

/// Name of the singleton GPU module that collects all CheddarGPU kernels.
inline constexpr StringLiteral kCheddarGpuModuleName = "cheddar_kernels";

/// Attaches NVPTX64 data layout specification to the GPU module so that 64-bit
/// memory accesses are properly aligned and lowered to 64-bit PTX memory ops
/// (ld.global.b64 / st.global.b64) rather than 32-bit pairs (ld.global.b32).
inline void attachNvptxDataLayout(gpu::GPUModuleOp gpuMod) {
  if (gpuMod->hasAttr(DLTIDialect::kDataLayoutAttrName)) return;
  MLIRContext* ctx = gpuMod.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto i128Type = IntegerType::get(ctx, 128);
  auto i64Vec2 = VectorType::get({2}, i64Type);
  SmallVector<DataLayoutEntryInterface> entries = {
      DataLayoutEntryAttr::get(
          StringAttr::get(ctx, DLTIDialect::kDataLayoutEndiannessKey),
          StringAttr::get(ctx, DLTIDialect::kDataLayoutEndiannessLittle)),
      DataLayoutEntryAttr::get(
          i64Type,
          DenseIntElementsAttr::get(i64Vec2, ArrayRef<int64_t>{64, 64})),
      DataLayoutEntryAttr::get(
          i128Type,
          DenseIntElementsAttr::get(i64Vec2, ArrayRef<int64_t>{128, 128})),
      DataLayoutEntryAttr::get(
          StringAttr::get(ctx, DLTIDialect::kDataLayoutLegalIntWidthsKey),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{16, 32, 64})),
      DataLayoutEntryAttr::get(IndexType::get(ctx),
                               IntegerAttr::get(i64Type, 64)),
  };
  gpuMod->setAttr(DLTIDialect::kDataLayoutAttrName,
                  DataLayoutSpecAttr::get(ctx, entries));
}

/// Finds or creates the singleton gpu.module named `kCheddarGpuModuleName`
/// in the given parent ModuleOp, and marks the parent module as a container
/// module.
inline gpu::GPUModuleOp getOrCreateGpuModule(
    ModuleOp module, StringRef moduleName = kCheddarGpuModuleName) {
  if (auto gpuMod = module.lookupSymbol<gpu::GPUModuleOp>(moduleName)) {
    attachNvptxDataLayout(gpuMod);
    return gpuMod;
  }
  OpBuilder b(&module.getBodyRegion().front(),
              module.getBodyRegion().front().begin());
  auto gpuMod = b.create<gpu::GPUModuleOp>(module.getLoc(), moduleName);
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  UnitAttr::get(module.getContext()));
  attachNvptxDataLayout(gpuMod);
  return gpuMod;
}

/// Finds or creates global memrefs for primes and Montgomery inverses inside
/// gpuMod.
inline std::pair<memref::GlobalOp, memref::GlobalOp>
getOrCreateDevicePrimeGlobals(gpu::GPUModuleOp gpuMod, const ParamsAttr& params,
                              uint32_t word) {
  auto primesGlobal = gpuMod.lookupSymbol<memref::GlobalOp>("__cheddar_primes");
  auto qInvGlobal = gpuMod.lookupSymbol<memref::GlobalOp>("__cheddar_q_inv");
  if (primesGlobal && qInvGlobal) {
    return {primesGlobal, qInvGlobal};
  }

  OpBuilder b(gpuMod.getBody(), gpuMod.getBody()->begin());
  Location loc = gpuMod.getLoc();
  Type intType = IntegerType::get(gpuMod.getContext(), word);
  size_t totalLimbs = params.getNumLimbs();

  SmallVector<APInt> primesVec;
  SmallVector<APInt> qInvVec;
  primesVec.reserve(totalLimbs);
  qInvVec.reserve(totalLimbs);

  for (size_t i = 0; i < totalLimbs; ++i) {
    uint64_t q = params.getPrimeAt(i);
    primesVec.push_back(APInt(word, q));
    uint64_t qInv =
        (word == 32)
            ? static_cast<uint64_t>(static_cast<uint32_t>(
                  cheddar_ref::MontArith<32>::ComputeQInv(q)))
            : static_cast<uint64_t>(cheddar_ref::MontArith<64>::ComputeQInv(q));
    qInvVec.push_back(APInt(word, qInv));
  }

  auto tensorType =
      RankedTensorType::get({static_cast<int64_t>(totalLimbs)}, intType);
  auto memrefType =
      MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);

  if (!primesGlobal) {
    auto primesDense = DenseElementsAttr::get(tensorType, primesVec);
    primesGlobal = memref::GlobalOp::create(
        b, loc, "__cheddar_primes", b.getStringAttr("private"), memrefType,
        primesDense, /*constant=*/true, /*alignment=*/IntegerAttr());
  }

  if (!qInvGlobal) {
    auto qInvDense = DenseElementsAttr::get(tensorType, qInvVec);
    qInvGlobal = memref::GlobalOp::create(
        b, loc, "__cheddar_q_inv", b.getStringAttr("private"), memrefType,
        qInvDense, /*constant=*/true, /*alignment=*/IntegerAttr());
  }

  return {primesGlobal, qInvGlobal};
}

/// Helper to get Value handles to the device prime globals inside a GPU kernel.
inline std::pair<Value, Value> getDevicePrimeGlobals(ImplicitLocOpBuilder& b,
                                                     uint32_t word,
                                                     size_t totalLimbs = 0) {
  MemRefType memrefType;
  if (auto gpuMod = b.getInsertionBlock()
                        ->getParentOp()
                        ->getParentOfType<gpu::GPUModuleOp>()) {
    if (auto g = gpuMod.lookupSymbol<memref::GlobalOp>("__cheddar_primes")) {
      memrefType = g.getType();
    }
  }
  if (!memrefType) {
    Type intType = IntegerType::get(b.getContext(), word);
    memrefType = MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);
  }
  Value primes = memref::GetGlobalOp::create(b, memrefType, "__cheddar_primes");
  Value qInv = memref::GetGlobalOp::create(b, memrefType, "__cheddar_q_inv");
  return {primes, qInv};
}

/// Finds or creates a memoized gpu.func kernel in `gpuMod`.
///
/// If a kernel named `kernelName` already exists in `gpuMod`, returns it
/// immediately. Otherwise:
/// 1. Creates `gpu.func @<kernelName>(<argTypes>) kernel` inside `gpuMod`.
/// 2. Creates an entry block with arguments matching `argTypes`.
/// 3. Calls `bodyBuilder(builder, loc, entryBlock)`.
/// 4. Ensures the entry block ends with `gpu.return`.
///
/// Memo key convention:
///   "cheddar_gpu_<op>_w<word>_deg<degree>[_<variant>]"
///   e.g.:
///     "cheddar_gpu_add_w64_deg65536"
///     "cheddar_gpu_copy_w64_deg65536"
///     "cheddar_gpu_ntt_fused_p1_w64_deg65536"
///     "cheddar_gpu_ntt_fused_p2_w64_deg65536"
inline gpu::GPUFuncOp getOrCreateKernel(
    gpu::GPUModuleOp gpuMod, StringRef kernelName, FunctionType kernelType,
    llvm::function_ref<void(OpBuilder& b, Location loc, Block* entryBlock)>
        bodyBuilder) {
  if (auto existing = gpuMod.lookupSymbol<gpu::GPUFuncOp>(kernelName)) {
    return existing;
  }
  OpBuilder b(gpuMod.getBody(), gpuMod.getBody()->end());
  Location loc = gpuMod.getLoc();
  auto kernelFunc = b.create<gpu::GPUFuncOp>(loc, kernelName, kernelType);
  kernelFunc.setKernel(true);

  Block* entryBlock = kernelFunc.getBody().empty()
                          ? kernelFunc.addEntryBlock()
                          : &kernelFunc.getBody().front();
  OpBuilder bodyB(entryBlock, entryBlock->begin());
  bodyBuilder(bodyB, loc, entryBlock);

  if (entryBlock->empty() ||
      !entryBlock->back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder termB(entryBlock, entryBlock->end());
    termB.create<gpu::ReturnOp>(loc);
  }
  return kernelFunc;
}

/// Finds or creates a memoized gpu.func kernel with workgroup shared memory in
/// `gpuMod`.
inline gpu::GPUFuncOp getOrCreateKernelWithSharedMem(
    gpu::GPUModuleOp gpuMod, StringRef kernelName, FunctionType kernelType,
    int64_t smemElements, Type smemElemType,
    llvm::function_ref<void(OpBuilder& b, Location loc, Block* entryBlock,
                            Value smem)>
        bodyBuilder) {
  if (auto existing = gpuMod.lookupSymbol<gpu::GPUFuncOp>(kernelName)) {
    return existing;
  }
  OpBuilder b(gpuMod.getBody(), gpuMod.getBody()->end());
  Location loc = gpuMod.getLoc();
  auto kernelFunc = b.create<gpu::GPUFuncOp>(loc, kernelName, kernelType);
  kernelFunc.setKernel(true);

  Block* entryBlock = kernelFunc.getBody().empty()
                          ? kernelFunc.addEntryBlock()
                          : &kernelFunc.getBody().front();

  Value smem;
  if (smemElements > 0) {
    auto addrSpace = gpu::AddressSpaceAttr::get(gpuMod.getContext(),
                                                gpu::AddressSpace::Workgroup);
    auto smemType = MemRefType::get({smemElements}, smemElemType,
                                    MemRefLayoutAttrInterface(), addrSpace);
    smem = kernelFunc.addWorkgroupAttribution(smemType, loc);
  }

  OpBuilder bodyB(entryBlock, entryBlock->begin());
  bodyBuilder(bodyB, loc, entryBlock, smem);

  if (entryBlock->empty() ||
      !entryBlock->back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder termB(entryBlock, entryBlock->end());
    termB.create<gpu::ReturnOp>(loc);
  }
  return kernelFunc;
}

/// Finds or creates a memref::GetGlobalOp in the enclosing function's entry
/// block so multiple operations in the same function reuse the single handle.
inline Value getOrCreateHostGetGlobal(OpBuilder& b, Location loc,
                                      memref::GlobalOp globalOp,
                                      Operation* currentOp) {
  auto parentFunc = currentOp->getParentOfType<FunctionOpInterface>();
  if (parentFunc && !parentFunc.getFunctionBody().empty()) {
    Block& entryBlock = parentFunc.getFunctionBody().front();
    for (Operation& op : entryBlock) {
      if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(op)) {
        if (getGlobal.getName() == globalOp.getSymName()) {
          return getGlobal.getResult();
        }
      }
    }
    OpBuilder entryB(&entryBlock, entryBlock.begin());
    return entryB
        .create<memref::GetGlobalOp>(loc, globalOp.getType(),
                                     globalOp.getSymName())
        .getResult();
  }
  return b
      .create<memref::GetGlobalOp>(loc, globalOp.getType(),
                                   globalOp.getSymName())
      .getResult();
}

/// Finds or creates a global memref for a specific prime index list.
inline memref::GlobalOp getOrCreatePrimeIndicesGlobal(
    ModuleOp module, ArrayRef<int64_t> indices) {
  std::string symName = "__cheddar_prime_indices";
  for (int64_t idx : indices) {
    symName += "_" + std::to_string(idx);
  }
  if (auto existing = module.lookupSymbol<memref::GlobalOp>(symName)) {
    return existing;
  }
  OpBuilder b(&module.getBodyRegion().front(),
              module.getBodyRegion().front().begin());
  Location loc = module.getLoc();
  Type indexType = b.getIndexType();
  int64_t count = indices.size();
  SmallVector<APInt> indicesVec;
  indicesVec.reserve(count);
  for (int64_t idx : indices) {
    indicesVec.push_back(APInt(64, idx));
  }
  auto tensorType = RankedTensorType::get({count}, indexType);
  auto denseAttr = DenseElementsAttr::get(tensorType, indicesVec);
  auto memrefType = MemRefType::get({count}, indexType);
  return memref::GlobalOp::create(b, loc, symName, b.getStringAttr("private"),
                                  memrefType, denseAttr, /*constant=*/true,
                                  /*alignment=*/IntegerAttr());
}

/// Finds or creates module-level host globals for NTT/INTT twiddles and scaling
/// tables.
template <unsigned W>
inline void getOrCreateModuleTwiddleGlobals(ModuleOp module,
                                            const ParamsAttr& paramsAttr,
                                            int64_t degree) {
  Type intType = IntegerType::get(module.getContext(), W);
  size_t totalLimbs = paramsAttr.getNumLimbs();

  std::string fwdName =
      llvm::formatv("__cheddar_twiddle_fwd_w{0}_deg{1}", W, degree).str();
  if (module.lookupSymbol<memref::GlobalOp>(fwdName)) return;

  std::string invName =
      llvm::formatv("__cheddar_twiddle_inv_w{0}_deg{1}", W, degree).str();
  std::string r2Name = llvm::formatv("__cheddar_r2_w{0}", W).str();
  std::string nInvName =
      llvm::formatv("__cheddar_n_inv_w{0}_deg{1}", W, degree).str();
  std::string nInvMontName =
      llvm::formatv("__cheddar_n_inv_mont_w{0}_deg{1}", W, degree).str();
  std::string dummyIndicesName = "__cheddar_dummy_indices";

  OpBuilder b(&module.getBodyRegion().front(),
              module.getBodyRegion().front().begin());
  Location loc = module.getLoc();

  const auto& tables = getCachedTables<W>(paramsAttr);

  // 1. Forward twiddles: memref<totalLimbs x degree x W>
  {
    SmallVector<APInt> twiddles;
    twiddles.reserve(totalLimbs * degree);
    for (size_t p = 0; p < totalLimbs; ++p) {
      for (int64_t i = 0; i < degree; ++i) {
        twiddles.push_back(APInt(W, tables.psi_rev_mont[p * degree + i]));
      }
    }
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(totalLimbs), degree}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, twiddles);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(totalLimbs), degree}, intType);
    memref::GlobalOp::create(b, loc, fwdName, b.getStringAttr("private"),
                             memrefType, denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr());
  }

  // 2. Inverse twiddles: memref<totalLimbs x degree x W>
  {
    SmallVector<APInt> twiddles;
    twiddles.reserve(totalLimbs * degree);
    for (size_t p = 0; p < totalLimbs; ++p) {
      for (int64_t i = 0; i < degree; ++i) {
        twiddles.push_back(APInt(W, tables.psi_inv_rev_mont[p * degree + i]));
      }
    }
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(totalLimbs), degree}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, twiddles);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(totalLimbs), degree}, intType);
    memref::GlobalOp::create(b, loc, invName, b.getStringAttr("private"),
                             memrefType, denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr());
  }

  // 3. R2 mod q: memref<totalLimbs x W>
  if (!module.lookupSymbol<memref::GlobalOp>(r2Name)) {
    SmallVector<APInt> r2Vec;
    r2Vec.reserve(totalLimbs);
    for (size_t p = 0; p < totalLimbs; ++p) {
      r2Vec.push_back(APInt(W, tables.R2_mod_q[p]));
    }
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(totalLimbs)}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, r2Vec);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);
    memref::GlobalOp::create(b, loc, r2Name, b.getStringAttr("private"),
                             memrefType, denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr());
  }

  // 4. N_inv: memref<totalLimbs x W>
  if (!module.lookupSymbol<memref::GlobalOp>(nInvName)) {
    SmallVector<APInt> nInvVec;
    nInvVec.reserve(totalLimbs);
    for (size_t p = 0; p < totalLimbs; ++p) {
      nInvVec.push_back(APInt(W, tables.N_inv[p]));
    }
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(totalLimbs)}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, nInvVec);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);
    memref::GlobalOp::create(b, loc, nInvName, b.getStringAttr("private"),
                             memrefType, denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr());
  }

  // 5. N_inv_mont: memref<totalLimbs x W>
  if (!module.lookupSymbol<memref::GlobalOp>(nInvMontName)) {
    SmallVector<APInt> nInvMontVec;
    nInvMontVec.reserve(totalLimbs);
    for (size_t p = 0; p < totalLimbs; ++p) {
      nInvMontVec.push_back(APInt(W, tables.N_inv_mont[p]));
    }
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(totalLimbs)}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, nInvMontVec);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);
    memref::GlobalOp::create(b, loc, nInvMontName, b.getStringAttr("private"),
                             memrefType, denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr());
  }

  // 6. Dummy indices: memref<1xindex>
  if (!module.lookupSymbol<memref::GlobalOp>(dummyIndicesName)) {
    Type indexType = b.getIndexType();
    SmallVector<APInt> dummyVec = {APInt(64, 0)};
    auto tensorType = RankedTensorType::get({1}, indexType);
    auto denseAttr = DenseElementsAttr::get(tensorType, dummyVec);
    auto memrefType = MemRefType::get({1}, indexType);
    memref::GlobalOp::create(b, loc, dummyIndicesName,
                             b.getStringAttr("private"), memrefType, denseAttr,
                             /*constant=*/true, /*alignment=*/IntegerAttr());
  }
}

/// Emits a gpu.launch_func operation calling `kernelFunc`.
inline gpu::LaunchFuncOp emitKernelLaunch(
    OpBuilder& b, Location loc, gpu::GPUFuncOp kernelFunc,
    gpu::KernelDim3 gridSize, gpu::KernelDim3 blockSize, ValueRange kernelArgs,
    Value dynamicSharedMemorySize = Value()) {
  return gpu::LaunchFuncOp::create(b, loc, kernelFunc, gridSize, blockSize,
                                   dynamicSharedMemorySize, kernelArgs);
}

/// Convenience helper: gets or creates the kernel in `module`'s GPU module
/// and launches it.
inline gpu::LaunchFuncOp emitKernelLaunch(
    OpBuilder& b, Location loc, ModuleOp module, StringRef kernelName,
    FunctionType kernelType, gpu::KernelDim3 gridSize,
    gpu::KernelDim3 blockSize, ValueRange kernelArgs,
    llvm::function_ref<void(OpBuilder& b, Location loc, Block* entryBlock)>
        bodyBuilder,
    Value dynamicSharedMemorySize = Value()) {
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);
  gpu::GPUFuncOp kernelFunc =
      getOrCreateKernel(gpuMod, kernelName, kernelType, bodyBuilder);
  return emitKernelLaunch(b, loc, kernelFunc, gridSize, blockSize, kernelArgs,
                          dynamicSharedMemorySize);
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_KERNELBUILDERS_H_
