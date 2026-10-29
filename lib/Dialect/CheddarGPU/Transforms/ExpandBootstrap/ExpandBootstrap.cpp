#include "lib/Dialect/CheddarGPU/Transforms/ExpandBootstrap/ExpandBootstrap.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Utils/CheddarRef/BootConsts.h"
#include "lib/Utils/CheddarRef/Encoder.h"
#include "lib/Utils/CheddarRef/ModSwitchTables.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_EXPANDBOOTSTRAP
#include "lib/Dialect/CheddarGPU/Transforms/ExpandBootstrap/ExpandBootstrap.h.inc"

namespace {

template <unsigned W>
cheddar_ref::Params<W> paramsFromAttr(ParamsAttr attr) {
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
  p.L = p.ter_primes.size() + p.main_primes.size();

  for (auto attrPair : attr.getLevelConfig()) {
    auto arr = cast<ArrayAttr>(attrPair);
    if (!arr.empty() && isa<ArrayAttr>(arr[0])) {
      arr = cast<ArrayAttr>(arr[0]);
    }
    int m = cast<IntegerAttr>(arr[0]).getInt();
    int t = cast<IntegerAttr>(arr[1]).getInt();
    p.level_config.push_back({m, t});
  }

  p.scale_info.log_default_scale = 40;
  p.scale_info.default_encryption_level = 13;
  p.scale_info.max_level =
      p.level_config.empty() ? 25 : static_cast<int>(p.level_config.size() - 1);

  return p;
}

template <typename Word>
Value getOrCreate1DGlobal(ImplicitLocOpBuilder& b, ModuleOp module,
                          StringRef name, ArrayRef<Word> data, Type intType) {
  auto globalOp = module.lookupSymbol<memref::GlobalOp>(name);
  if (!globalOp) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    SmallVector<APInt> elements;
    elements.reserve(data.size());
    unsigned width = intType.getIntOrFloatBitWidth();
    for (auto val : data) {
      elements.push_back(APInt(
          width,
          static_cast<uint64_t>(static_cast<std::make_unsigned_t<Word>>(val))));
    }
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(data.size())}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, elements);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(data.size())}, intType);
    memref::GlobalOp::create(b, name, b.getStringAttr("private"), memrefType,
                             denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr{});
  }
  auto memrefType =
      MemRefType::get({static_cast<int64_t>(data.size())}, intType);
  return memref::GetGlobalOp::create(b, memrefType, name);
}

struct CtValue {
  Value bx;
  Value ax;
  Value rx = nullptr;  // non-null only for degree-2 ciphertexts
};

Value sliceToLevel(ImplicitLocOpBuilder& b, Value val, int target_level,
                   int64_t degree, Type intType) {
  auto type = cast<MemRefType>(val.getType());
  int64_t current_limbs = type.getDimSize(0);
  int64_t target_limbs = target_level + 1;
  if (current_limbs == target_limbs) {
    return val;
  }
  assert(current_limbs > target_limbs);
  auto targetType = MemRefType::get({target_limbs, degree}, intType);
  return memref::ReinterpretCastOp::create(
      b, targetType, val, /*offset=*/0,
      /*sizes=*/ArrayRef<int64_t>{target_limbs, degree},
      /*strides=*/ArrayRef<int64_t>{degree, 1});
}

CtValue sliceCtToLevel(ImplicitLocOpBuilder& b, CtValue ct, int target_level,
                       int64_t degree, Type intType) {
  Value bx = sliceToLevel(b, ct.bx, target_level, degree, intType);
  Value ax = sliceToLevel(b, ct.ax, target_level, degree, intType);
  Value rx =
      ct.rx ? sliceToLevel(b, ct.rx, target_level, degree, intType) : nullptr;
  return CtValue{bx, ax, rx};
}

CtValue evalAXYPBZ(ImplicitLocOpBuilder& b, ModuleOp module,
                   const cheddar_ref::AXYPBZConsts<64>& eval, CtValue x,
                   CtValue y, std::optional<CtValue> z, ValueRange evkBx,
                   ValueRange evkAx, int64_t degree, Type intType,
                   StringRef prefix,
                   std::optional<CtValue> dest = std::nullopt) {
  int working_level = eval.has_z
                          ? std::min({eval.x_level, eval.y_level, eval.z_level})
                          : std::min(eval.x_level, eval.y_level);
  auto workingType = MemRefType::get({working_level + 1, degree}, intType);

  x = sliceCtToLevel(b, x, working_level, degree, intType);
  y = sliceCtToLevel(b, y, working_level, degree, intType);
  if (z.has_value()) {
    z = sliceCtToLevel(b, *z, working_level, degree, intType);
  }

  Value x_bx = x.bx;
  Value x_ax = x.ax;
  if (eval.has_a) {
    Value a_cst = getOrCreate1DGlobal(b, module, (prefix + "_a").str(),
                                      ArrayRef<uint64_t>(eval.a_cx), intType);
    Value ax_bx = memref::AllocOp::create(b, workingType);
    Value ax_ax = memref::AllocOp::create(b, workingType);
    CPAccum2Op::create(b, ax_bx, ax_ax, ValueRange{x_bx}, ValueRange{x_ax},
                       ValueRange{a_cst}, Form::Eval,
                       /*ct_forms=*/ArrayAttr{}, /*common_forms=*/ArrayAttr{},
                       /*out_limb_range=*/DenseI64ArrayAttr{},
                       /*prime_offset=*/IntegerAttr{},
                       /*num_aux=*/IntegerAttr{});
    x_bx = ax_bx;
    x_ax = ax_ax;
  }

  Value t_bx = memref::AllocOp::create(b, workingType);
  Value t_ax = memref::AllocOp::create(b, workingType);
  Value t_rx = memref::AllocOp::create(b, workingType);
  TensorOp::create(b, t_bx, t_ax, t_rx, x_bx, x_ax, y.bx, y.ax, Form::Eval,
                   Form::Eval, Form::Eval,
                   /*out_limb_range=*/DenseI64ArrayAttr{});

  if (eval.has_b) {
    Value b_cst = getOrCreate1DGlobal(b, module, (prefix + "_b").str(),
                                      ArrayRef<uint64_t>(eval.b_cx), intType);
    if (eval.has_z) {
      assert(z.has_value() && "z must be provided when has_z is true");
      Value bz_bx = memref::AllocOp::create(b, workingType);
      Value bz_ax = memref::AllocOp::create(b, workingType);
      CPAccum2Op::create(b, bz_bx, bz_ax, ValueRange{z->bx}, ValueRange{z->ax},
                         ValueRange{b_cst}, Form::Eval,
                         /*ct_forms=*/ArrayAttr{},
                         /*common_forms=*/ArrayAttr{},
                         /*out_limb_range=*/DenseI64ArrayAttr{},
                         /*prime_offset=*/IntegerAttr{},
                         /*num_aux=*/IntegerAttr{});
      Add2Op::create(b, t_bx, t_ax, t_bx, t_ax, bz_bx, bz_ax, Form::Eval,
                     Form::Eval, Form::Eval, DenseI64ArrayAttr{},
                     DenseI64ArrayAttr{}, DenseI64ArrayAttr{}, IntegerAttr{});
    } else {
      AddConstOp::create(b, t_bx, t_bx, b_cst, Form::Eval, Form::Eval,
                         DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
    }
  }

  int final_level = working_level - 1;
  Value res_bx, res_ax;
  if (dest.has_value()) {
    res_bx = dest->bx;
    res_ax = dest->ax;
  } else {
    auto finalType = MemRefType::get({final_level + 1, degree}, intType);
    res_bx = memref::AllocOp::create(b, finalType);
    res_ax = memref::AllocOp::create(b, finalType);
  }

  RelinearizeRescaleOp::create(b, res_bx, res_ax, t_bx, t_ax, t_rx, evkBx,
                               evkAx, working_level, Form::Eval, Form::Eval);
  return CtValue{res_bx, res_ax, nullptr};
}

CtValue evalPolyNode(ImplicitLocOpBuilder& b, ModuleOp module,
                     std::shared_ptr<cheddar_ref::RefEvalPolyNode<64>> node,
                     const std::map<int, CtValue>& basis_cts, ValueRange evkBx,
                     ValueRange evkAx, int64_t degree, Type intType,
                     const std::string& prefix) {
  int working_level = node->GetTargetLevel();
  if (node->GetDoRescale()) {
    working_level += 1;
  }
  auto workingType = MemRefType::get({working_level + 1, degree}, intType);

  if (node->IsLeaf()) {
    Value accum_bx = memref::AllocOp::create(b, workingType);
    Value accum_ax = memref::AllocOp::create(b, workingType);
    bool first = true;
    for (const auto& [base_deg, cst_vec] : node->GetLeafConstants()) {
      if (base_deg == 0) continue;
      CtValue base_ct = sliceCtToLevel(b, basis_cts.at(base_deg), working_level,
                                       degree, intType);
      Value cst = getOrCreate1DGlobal(
          b, module, prefix + "_leaf_c" + std::to_string(base_deg),
          ArrayRef<uint64_t>(cst_vec), intType);
      Value term_bx = memref::AllocOp::create(b, workingType);
      Value term_ax = memref::AllocOp::create(b, workingType);
      CPAccum2Op::create(b, term_bx, term_ax, ValueRange{base_ct.bx},
                         ValueRange{base_ct.ax}, ValueRange{cst}, Form::Eval,
                         /*ct_forms=*/ArrayAttr{},
                         /*common_forms=*/ArrayAttr{},
                         /*out_limb_range=*/DenseI64ArrayAttr{},
                         /*prime_offset=*/IntegerAttr{},
                         /*num_aux=*/IntegerAttr{});
      if (first) {
        CopyOp::create(b, accum_bx, term_bx, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{});
        CopyOp::create(b, accum_ax, term_ax, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{});
        first = false;
      } else {
        Add2Op::create(b, accum_bx, accum_ax, accum_bx, accum_ax, term_bx,
                       term_ax, Form::Eval, Form::Eval, Form::Eval,
                       DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{}, IntegerAttr{});
      }
    }
    if (node->GetLeafConstants().count(0)) {
      Value cst0 = getOrCreate1DGlobal(
          b, module, prefix + "_leaf_c0",
          ArrayRef<uint64_t>(node->GetLeafConstants().at(0)), intType);
      AddConstOp::create(b, accum_bx, accum_bx, cst0, Form::Eval, Form::Eval,
                         DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
    }
    if (node->GetDoRescale()) {
      auto targetType =
          MemRefType::get({node->GetTargetLevel() + 1, degree}, intType);
      Value res_bx = memref::AllocOp::create(b, targetType);
      Value res_ax = memref::AllocOp::create(b, targetType);
      RescaleOp::create(b, res_bx, accum_bx, working_level, Form::Eval,
                        Form::Eval);
      RescaleOp::create(b, res_ax, accum_ax, working_level, Form::Eval,
                        Form::Eval);
      return CtValue{res_bx, res_ax, nullptr};
    }
    return CtValue{accum_bx, accum_ax, nullptr};
  }

  // Interior / middle node
  int split_deg = node->GetSplitDegree();
  CtValue split = sliceCtToLevel(b, basis_cts.at(split_deg), working_level,
                                 degree, intType);

  Value accum_bx, accum_ax, accum_rx;
  bool accum_has_rx = false;

  if (node->GetHigh() != nullptr) {
    CtValue high_val =
        evalPolyNode(b, module, node->GetHigh(), basis_cts, evkBx, evkAx,
                     degree, intType, prefix + "_h");
    accum_bx = memref::AllocOp::create(b, workingType);
    accum_ax = memref::AllocOp::create(b, workingType);
    accum_rx = memref::AllocOp::create(b, workingType);
    TensorOp::create(b, accum_bx, accum_ax, accum_rx, high_val.bx, high_val.ax,
                     split.bx, split.ax, Form::Eval, Form::Eval, Form::Eval,
                     /*out_limb_range=*/DenseI64ArrayAttr{});
    accum_has_rx = true;
  } else if (node->IsHighConstant()) {
    Value high_cst = getOrCreate1DGlobal(
        b, module, prefix + "_high_cst",
        ArrayRef<uint64_t>(node->GetHighConstant()), intType);
    accum_bx = memref::AllocOp::create(b, workingType);
    accum_ax = memref::AllocOp::create(b, workingType);
    CPAccum2Op::create(b, accum_bx, accum_ax, ValueRange{split.bx},
                       ValueRange{split.ax}, ValueRange{high_cst}, Form::Eval,
                       /*ct_forms=*/ArrayAttr{},
                       /*common_forms=*/ArrayAttr{},
                       /*out_limb_range=*/DenseI64ArrayAttr{},
                       /*prime_offset=*/IntegerAttr{},
                       /*num_aux=*/IntegerAttr{});
    accum_has_rx = false;
  }

  if (node->GetLow() != nullptr) {
    if (node->GetLow()->IsLeaf()) {
      for (const auto& [base_deg, cst_vec] :
           node->GetLow()->GetLeafConstants()) {
        if (base_deg == 0) continue;
        CtValue base_ct = sliceCtToLevel(b, basis_cts.at(base_deg),
                                         working_level, degree, intType);
        Value cst = getOrCreate1DGlobal(
            b, module, prefix + "_low_leaf_c" + std::to_string(base_deg),
            ArrayRef<uint64_t>(cst_vec), intType);
        Value term_bx = memref::AllocOp::create(b, workingType);
        Value term_ax = memref::AllocOp::create(b, workingType);
        CPAccum2Op::create(b, term_bx, term_ax, ValueRange{base_ct.bx},
                           ValueRange{base_ct.ax}, ValueRange{cst}, Form::Eval,
                           /*ct_forms=*/ArrayAttr{},
                           /*common_forms=*/ArrayAttr{},
                           /*out_limb_range=*/DenseI64ArrayAttr{},
                           /*prime_offset=*/IntegerAttr{},
                           /*num_aux=*/IntegerAttr{});
        Add2Op::create(b, accum_bx, accum_ax, accum_bx, accum_ax, term_bx,
                       term_ax, Form::Eval, Form::Eval, Form::Eval,
                       DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{}, IntegerAttr{});
      }
      if (node->GetLow()->GetLeafConstants().count(0)) {
        Value cst0 = getOrCreate1DGlobal(
            b, module, prefix + "_low_leaf_c0",
            ArrayRef<uint64_t>(node->GetLow()->GetLeafConstants().at(0)),
            intType);
        AddConstOp::create(b, accum_bx, accum_bx, cst0, Form::Eval, Form::Eval,
                           DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
      }
    } else {
      CtValue low_val =
          evalPolyNode(b, module, node->GetLow(), basis_cts, evkBx, evkAx,
                       degree, intType, prefix + "_l");
      Add2Op::create(b, accum_bx, accum_ax, accum_bx, accum_ax, low_val.bx,
                     low_val.ax, Form::Eval, Form::Eval, Form::Eval,
                     DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                     DenseI64ArrayAttr{}, IntegerAttr{});
      if (accum_has_rx && low_val.rx != nullptr) {
        AddOp::create(b, accum_rx, accum_rx, low_val.rx, Form::Eval, Form::Eval,
                      Form::Eval, DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                      DenseI64ArrayAttr{});
      } else if (low_val.rx != nullptr) {
        accum_rx = low_val.rx;
        accum_has_rx = true;
      }
    }
  } else if (node->IsLowConstant() && !node->IsLowZero()) {
    Value low_cst = getOrCreate1DGlobal(
        b, module, prefix + "_low_cst",
        ArrayRef<uint64_t>(node->GetLowConstant()), intType);
    AddConstOp::create(b, accum_bx, accum_bx, low_cst, Form::Eval, Form::Eval,
                       DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
  }

  if (node->GetDoRescale()) {
    auto targetType =
        MemRefType::get({node->GetTargetLevel() + 1, degree}, intType);
    Value res_bx = memref::AllocOp::create(b, targetType);
    Value res_ax = memref::AllocOp::create(b, targetType);
    if (accum_has_rx) {
      RelinearizeRescaleOp::create(b, res_bx, res_ax, accum_bx, accum_ax,
                                   accum_rx, evkBx, evkAx, working_level,
                                   Form::Eval, Form::Eval);
    } else {
      RescaleOp::create(b, res_bx, accum_bx, working_level, Form::Eval,
                        Form::Eval);
      RescaleOp::create(b, res_ax, accum_ax, working_level, Form::Eval,
                        Form::Eval);
    }
    return CtValue{res_bx, res_ax, nullptr};
  }

  return CtValue{accum_bx, accum_ax, accum_has_rx ? accum_rx : nullptr};
}

void emitLevelDown(ImplicitLocOpBuilder& b, ModuleOp module,
                   const cheddar_ref::BootConsts<64>& boot_consts, Value inBx,
                   Value inAx, int fromLevel, int toLevel, Value outBx,
                   Value outAx, int64_t degree, Type intType) {
  if (fromLevel <= toLevel) {
    if (inBx != outBx) {
      CopyOp::create(b, outBx, inBx, DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
      CopyOp::create(b, outAx, inAx, DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
    }
    return;
  }
  Value curBx = inBx;
  Value curAx = inAx;
  for (int lvl = fromLevel; lvl >= toLevel + 1; --lvl) {
    Value lvl_cst = getOrCreate1DGlobal(
        b, module, "__cheddar_level_down_const_l" + std::to_string(lvl),
        ArrayRef<uint64_t>(boot_consts.GetLevelDownConsts().at(lvl)), intType);
    auto multType = MemRefType::get({lvl + 1, degree}, intType);
    Value mult_bx = memref::AllocOp::create(b, multType);
    Value mult_ax = memref::AllocOp::create(b, multType);
    CPAccum2Op::create(b, mult_bx, mult_ax, ValueRange{curBx},
                       ValueRange{curAx}, ValueRange{lvl_cst}, Form::Eval,
                       /*ct_forms=*/ArrayAttr{}, /*common_forms=*/ArrayAttr{},
                       /*out_limb_range=*/DenseI64ArrayAttr{},
                       /*prime_offset=*/IntegerAttr{},
                       /*num_aux=*/IntegerAttr{});
    if (lvl == toLevel + 1) {
      RescaleOp::create(b, outBx, mult_bx, lvl, Form::Eval, Form::Eval);
      RescaleOp::create(b, outAx, mult_ax, lvl, Form::Eval, Form::Eval);
    } else {
      auto nextType = MemRefType::get({lvl, degree}, intType);
      Value next_bx = memref::AllocOp::create(b, nextType);
      Value next_ax = memref::AllocOp::create(b, nextType);
      RescaleOp::create(b, next_bx, mult_bx, lvl, Form::Eval, Form::Eval);
      RescaleOp::create(b, next_ax, mult_ax, lvl, Form::Eval, Form::Eval);
      curBx = next_bx;
      curAx = next_ax;
    }
  }
}

struct BootConstsEntry {
  cheddar_ref::Params<64> params;
  cheddar_ref::Tables<64> tables;
  std::unique_ptr<cheddar_ref::Encoder<64>> encoder;
  std::unique_ptr<cheddar_ref::BootConsts<64>> bootConsts;
};

struct BootConstsCache {
  DenseMap<ParamsAttr, std::unique_ptr<BootConstsEntry>> entries;

  cheddar_ref::BootConsts<64>* getBootConsts(
      ParamsAttr paramsAttr, const cheddar_ref::Params<64>& params) {
    auto it = entries.find(paramsAttr);
    if (it != entries.end()) return it->second->bootConsts.get();
    auto entry = std::make_unique<BootConstsEntry>();
    entry->params = params;
    entry->tables =
        cheddar_ref::Tables<64>::Generate(entry->params, /*max_limbs=*/0);
    entry->encoder = std::make_unique<cheddar_ref::Encoder<64>>(entry->params,
                                                                entry->tables);
    entry->bootConsts = std::make_unique<cheddar_ref::BootConsts<64>>(
        entry->params, entry->tables, *entry->encoder);
    auto* ptr = entry->bootConsts.get();
    entries[paramsAttr] = std::move(entry);
    return ptr;
  }
};

struct ExpandEvalModPattern : public OpRewritePattern<EvalModOp> {
  ExpandEvalModPattern(MLIRContext* context, BootConstsCache& cache)
      : OpRewritePattern<EvalModOp>(context), cache_(cache) {}

  LogicalResult matchAndRewrite(EvalModOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    ParamsAttr paramsAttr;
    if (op.getParams().has_value()) {
      paramsAttr = op.getParams().value();
    } else {
      paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    }
    if (!paramsAttr) {
      return op.emitOpError("missing 'cheddar_gpu.params' attribute on module");
    }

    auto params = paramsFromAttr<64>(paramsAttr);
    const auto& boot_consts = *cache_.getBootConsts(paramsAttr, params);

    int start_level = boot_consts.GetScalars().eval_mod_start_level;
    int64_t relinOffset = op.getRelinEvkByteOffset();
    std::string funcName;
    if (relinOffset != 0) {
      funcName = "__cheddar_eval_mod_l" + std::to_string(start_level) + "_off" +
                 std::to_string(relinOffset);
    } else {
      funcName = "__cheddar_eval_mod_l" + std::to_string(start_level);
    }

    auto funcOp = module.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());

      Type outType = op.getOutBx().getType();
      Type inType = op.getInBx().getType();
      Type slabType = op.getEvkSlab().getType();
      auto funcType = rewriter.getFunctionType(
          {outType, outType, inType, inType, slabType}, {});

      funcOp = func::FuncOp::create(rewriter, loc, funcName, funcType);
      funcOp.setPrivate();

      Block* entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      ImplicitLocOpBuilder b(loc, rewriter);

      Value func_out_bx = entryBlock->getArgument(0);
      Value func_out_ax = entryBlock->getArgument(1);
      Value func_in_bx = entryBlock->getArgument(2);
      Value func_in_ax = entryBlock->getArgument(3);
      Value func_evk_slab = entryBlock->getArgument(4);

      int64_t numQ = op.getRelinEvkNumQ();
      int64_t beta = op.getRelinEvkBeta();
      int64_t alpha = paramsAttr.getAlpha();
      int64_t degree = 1LL << paramsAttr.getLogN();
      Type intType = b.getI64Type();
      int64_t polyBytes = (numQ + alpha) * degree * sizeof(uint64_t);
      auto evkDigitType = MemRefType::get({numQ + alpha, degree}, intType);

      SmallVector<Value> evkBx, evkAx;
      evkBx.reserve(beta);
      evkAx.reserve(beta);
      for (int64_t d = 0; d < beta; ++d) {
        int64_t bxOffset = relinOffset + d * 2 * polyBytes;
        int64_t axOffset = bxOffset + polyBytes;
        Value bxOffVal = arith::ConstantIndexOp::create(b, bxOffset);
        Value axOffVal = arith::ConstantIndexOp::create(b, axOffset);
        Value bxView = memref::ViewOp::create(b, evkDigitType, func_evk_slab,
                                              bxOffVal, ValueRange{});
        Value axView = memref::ViewOp::create(b, evkDigitType, func_evk_slab,
                                              axOffVal, ValueRange{});
        evkBx.push_back(bxView);
        evkAx.push_back(axView);
      }

      // Step 0: Add initial constant to bx
      Value init_cst = getOrCreate1DGlobal(
          b, module, "__cheddar_eval_mod_initial_const",
          ArrayRef<uint64_t>(boot_consts.GetEvalModInitialConst()), intType);
      auto inMemRefType = cast<MemRefType>(inType);
      Value t0_bx = memref::AllocOp::create(b, inMemRefType);
      Value t0_ax = memref::AllocOp::create(b, inMemRefType);
      AddConstOp::create(b, t0_bx, func_in_bx, init_cst, Form::Eval, Form::Eval,
                         DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
      CopyOp::create(b, t0_ax, func_in_ax, DenseI64ArrayAttr{},
                     DenseI64ArrayAttr{});

      // Step 1: Evaluate basis map
      const auto& basis_map = boot_consts.GetBasisMap();
      const auto& basis_eval = basis_map.GetBasisEval();

      std::map<int, CtValue> basis_cts;
      basis_cts[1] = CtValue{t0_bx, t0_ax, nullptr};

      for (const auto& [base_degree, eval] : basis_eval) {
        int left_degree = basis_map.SplitBaseDegree(base_degree);
        int right_degree = base_degree - left_degree;
        int sub_degree = left_degree - right_degree;

        CtValue left = sliceCtToLevel(b, basis_cts.at(left_degree),
                                      eval.x_level, degree, intType);
        CtValue right = sliceCtToLevel(b, basis_cts.at(right_degree),
                                       eval.y_level, degree, intType);
        std::optional<CtValue> sub = std::nullopt;
        if (eval.has_z && sub_degree != 0) {
          sub = sliceCtToLevel(b, basis_cts.at(sub_degree), eval.z_level,
                               degree, intType);
        }

        CtValue new_base = evalAXYPBZ(
            b, module, eval, left, right, sub, evkBx, evkAx, degree, intType,
            "__cheddar_basis_" + std::to_string(base_degree));
        basis_cts[base_degree] = new_base;
      }

      // Step 2: Evaluate Chebyshev polynomial
      CtValue poly_res =
          evalPolyNode(b, module, boot_consts.GetEvalPolyRoot(), basis_cts,
                       evkBx, evkAx, degree, intType, "__cheddar_eval_poly");

      // Step 3: Double angle steps
      const auto& double_angle = boot_consts.GetDoubleAngleConsts();
      CtValue cur_res = poly_res;
      for (size_t i = 0; i < double_angle.size(); ++i) {
        bool is_last = (i == double_angle.size() - 1);
        std::optional<CtValue> dest = std::nullopt;
        if (is_last) {
          dest = CtValue{func_out_bx, func_out_ax, nullptr};
        }
        cur_res =
            evalAXYPBZ(b, module, double_angle[i], cur_res, cur_res,
                       std::nullopt, evkBx, evkAx, degree, intType,
                       "__cheddar_eval_mod_da_" + std::to_string(i), dest);
      }

      func::ReturnOp::create(b);
    }

    ImplicitLocOpBuilder b(loc, rewriter);
    func::CallOp::create(b, funcOp,
                         ValueRange{op.getOutBx(), op.getOutAx(), op.getInBx(),
                                    op.getInAx(), op.getEvkSlab()});
    rewriter.eraseOp(op);
    return success();
  }

 private:
  BootConstsCache& cache_;
};

struct ExpandBootPattern : public OpRewritePattern<BootOp> {
  ExpandBootPattern(MLIRContext* context, BootConstsCache& cache)
      : OpRewritePattern<BootOp>(context), cache_(cache) {}

  LogicalResult matchAndRewrite(BootOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    ParamsAttr paramsAttr;
    if (op.getParams().has_value()) {
      paramsAttr = op.getParams().value();
    } else {
      paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    }
    if (!paramsAttr) {
      return op.emitOpError("missing 'cheddar_gpu.params' attribute on module");
    }

    auto params = paramsFromAttr<64>(paramsAttr);
    const auto& boot_consts = *cache_.getBootConsts(paramsAttr, params);

    int64_t numSlots = op.getNumSlots();
    int64_t degree = 1LL << paramsAttr.getLogN();
    int64_t halfDegree = degree / 2;
    bool fullSlot = (numSlots == halfDegree);
    Type intType = rewriter.getI64Type();

    int64_t inLimbs = cast<MemRefType>(op.getInBx().getType()).getDimSize(0);
    int64_t outLimbs = cast<MemRefType>(op.getOutBx().getType()).getDimSize(0);
    int64_t inLevel = inLimbs - 1;
    int64_t outLevel = outLimbs - 1;

    std::string bootFuncName = "__cheddar_boot_s" + std::to_string(numSlots);
    auto bootFuncOp = module.lookupSymbol<func::FuncOp>(bootFuncName);
    if (!bootFuncOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());

      auto outBootType = MemRefType::get({11, degree}, intType);
      auto inBootType = MemRefType::get({1, degree}, intType);
      Type slabType = op.getConstsSlab().getType();
      auto funcType =
          rewriter.getFunctionType({outBootType, outBootType, inBootType,
                                    inBootType, slabType, slabType},
                                   {});

      bootFuncOp = func::FuncOp::create(rewriter, loc, bootFuncName, funcType);
      bootFuncOp.setPrivate();

      Block* entryBlock = bootFuncOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      ImplicitLocOpBuilder b(loc, rewriter);

      Value func_out_bx = entryBlock->getArgument(0);
      Value func_out_ax = entryBlock->getArgument(1);
      Value func_in_bx = entryBlock->getArgument(2);
      Value func_in_ax = entryBlock->getArgument(3);
      Value func_consts_slab = entryBlock->getArgument(4);
      Value func_evk_slab = entryBlock->getArgument(5);

      const auto& ltPlan = boot_consts.GetLinearTransformPlan(numSlots);
      const auto& evkPlan = boot_consts.GetBootEvkPlan(numSlots);

      auto viewEvk = [&](const cheddar_ref::BootEvkSpec& spec) {
        int64_t polyBytes = static_cast<int64_t>(spec.num_q + spec.num_aux) *
                            degree * sizeof(uint64_t);
        auto evkDigitType =
            MemRefType::get({spec.num_q + spec.num_aux, degree}, intType);
        SmallVector<Value> bxList, axList;
        bxList.reserve(spec.beta);
        axList.reserve(spec.beta);
        for (int64_t d = 0; d < spec.beta; ++d) {
          int64_t bxOffset = spec.slab_byte_offset + d * 2 * polyBytes;
          int64_t axOffset = bxOffset + polyBytes;
          Value bxOffVal = arith::ConstantIndexOp::create(b, bxOffset);
          Value axOffVal = arith::ConstantIndexOp::create(b, axOffset);
          Value bxView = memref::ViewOp::create(b, evkDigitType, func_evk_slab,
                                                bxOffVal, ValueRange{});
          Value axView = memref::ViewOp::create(b, evkDigitType, func_evk_slab,
                                                axOffVal, ValueRange{});
          bxList.push_back(bxView);
          axList.push_back(axView);
        }
        return std::make_pair(bxList, axList);
      };

      auto findEvkSpec = [&](cheddar_ref::BootEvkKind kind,
                             int rotIdx =
                                 0) -> const cheddar_ref::BootEvkSpec& {
        for (const auto& s : evkPlan.evks) {
          if (s.kind == kind && (kind != cheddar_ref::BootEvkKind::kRot ||
                                 s.rot_index == rotIdx)) {
            return s;
          }
        }
        llvm_unreachable("Required EVK spec not found in evkPlan");
      };

      // Stage 0: ScaleUp
      Value scaleupConst = getOrCreate1DGlobal(
          b, module, "__cheddar_scaleup_const",
          ArrayRef<uint64_t>(boot_consts.GetScaleupConst()), intType);
      auto s0Type = MemRefType::get({1, degree}, intType);
      Value s0_bx = memref::AllocOp::create(b, s0Type);
      Value s0_ax = memref::AllocOp::create(b, s0Type);
      CPAccum2Op::create(b, s0_bx, s0_ax, ValueRange{func_in_bx},
                         ValueRange{func_in_ax}, ValueRange{scaleupConst},
                         Form::Eval, /*ct_forms=*/ArrayAttr{},
                         /*common_forms=*/ArrayAttr{},
                         /*out_limb_range=*/DenseI64ArrayAttr{},
                         /*prime_offset=*/IntegerAttr{},
                         /*num_aux=*/IntegerAttr{});

      // Stage 1a: ModUpToMax
      const auto& dtsSpec = findEvkSpec(cheddar_ref::BootEvkKind::kDts);
      auto dtsEvk = viewEvk(dtsSpec);

      Value dts_bx = memref::AllocOp::create(b, s0Type);
      Value dts_ax = memref::AllocOp::create(b, s0Type);
      KeySwitchOp::create(b, dts_bx, dts_ax, s0_bx, s0_ax, dtsEvk.first,
                          dtsEvk.second, -1, Form::Eval, Form::Eval);

      Value modMaxIntt = getOrCreate1DGlobal(
          b, module, "__cheddar_mod_max_intt_const",
          ArrayRef<uint64_t>(boot_consts.GetModMaxInttConst()), intType);
      auto pseudoModUp =
          cheddar_ref::ModSwitchTables<64>::GeneratePseudoModUp(params, 25);
      Value pProdL25 =
          getOrCreate1DGlobal(b, module, "__cheddar_pseudo_mod_up_p_prod_l25",
                              ArrayRef<uint64_t>(pseudoModUp.p_prod), intType);

      INTTEpilogueAttr epilogueAttr =
          INTTEpilogueAttr::get(b.getContext(), INTTEpilogue::MultConst);

      Value c_bx = memref::AllocOp::create(b, s0Type);
      INTTOp::create(b, c_bx, dts_bx, modMaxIntt,
                     FormAttr::get(b.getContext(), Form::Eval),
                     FormAttr::get(b.getContext(), Form::Coeff),
                     /*montgomery_conversion=*/b.getBoolAttr(true),
                     epilogueAttr,
                     /*in_limb_range=*/DenseI64ArrayAttr{},
                     /*out_limb_range=*/DenseI64ArrayAttr{},
                     /*prime_range=*/DenseI64ArrayAttr{});

      auto l25Type = MemRefType::get({26, degree}, intType);
      Value tmp_bx = memref::AllocOp::create(b, l25Type);
      ModUpToMaxOp::create(b, tmp_bx, c_bx, Form::Coeff, Form::Coeff,
                           /*out_limb_range=*/DenseI64ArrayAttr{},
                           /*prime_offset=*/b.getI64IntegerAttr(0),
                           /*num_aux=*/b.getI64IntegerAttr(0));

      NTTOp::create(b, tmp_bx, tmp_bx,
                    FormAttr::get(b.getContext(), Form::Coeff),
                    FormAttr::get(b.getContext(), Form::Eval),
                    /*montgomery_conversion=*/b.getBoolAttr(true));

      Value c_ax = memref::AllocOp::create(b, s0Type);
      INTTOp::create(b, c_ax, dts_ax, modMaxIntt,
                     FormAttr::get(b.getContext(), Form::Eval),
                     FormAttr::get(b.getContext(), Form::Coeff),
                     /*montgomery_conversion=*/b.getBoolAttr(true),
                     epilogueAttr,
                     /*in_limb_range=*/DenseI64ArrayAttr{},
                     /*out_limb_range=*/DenseI64ArrayAttr{},
                     /*prime_range=*/DenseI64ArrayAttr{});

      int64_t alpha = paramsAttr.getAlpha();
      auto qpType = MemRefType::get({26 + alpha, degree}, intType);
      Value tmp_ax = memref::AllocOp::create(b, qpType);
      ModUpToMaxOp::create(b, tmp_ax, c_ax, Form::Coeff, Form::Coeff,
                           /*out_limb_range=*/DenseI64ArrayAttr{},
                           /*prime_offset=*/b.getI64IntegerAttr(0),
                           /*num_aux=*/b.getI64IntegerAttr(alpha));

      NTTOp::create(b, tmp_ax, tmp_ax,
                    FormAttr::get(b.getContext(), Form::Coeff),
                    FormAttr::get(b.getContext(), Form::Eval),
                    /*montgomery_conversion=*/b.getBoolAttr(true));

      const auto& stdSpec = findEvkSpec(cheddar_ref::BootEvkKind::kStd);
      auto stdEvk = viewEvk(stdSpec);

      Value std_prod_bx = memref::AllocOp::create(b, qpType);
      Value std_prod_ax = memref::AllocOp::create(b, qpType);
      MulOp::create(b, std_prod_bx, stdEvk.first[0], tmp_ax, Form::Eval,
                    Form::Eval, Form::Eval, DenseI64ArrayAttr{},
                    DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                    /*num_aux=*/b.getI64IntegerAttr(alpha));
      MulOp::create(b, std_prod_ax, stdEvk.second[0], tmp_ax, Form::Eval,
                    Form::Eval, Form::Eval, DenseI64ArrayAttr{},
                    DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                    /*num_aux=*/b.getI64IntegerAttr(alpha));

      CPAccumAddOp::create(
          b, std_prod_bx, std_prod_bx, ValueRange{tmp_bx}, ValueRange{pProdL25},
          FormAttr::get(b.getContext(), Form::Eval),
          FormAttr::get(b.getContext(), Form::Eval), /*ct_forms=*/ArrayAttr{},
          /*common_forms=*/ArrayAttr{}, b.getDenseI64ArrayAttr({0, 26}),
          /*prime_offset=*/IntegerAttr{}, /*num_aux=*/IntegerAttr{});

      Value mup_bx = memref::AllocOp::create(b, l25Type);
      Value mup_ax = memref::AllocOp::create(b, l25Type);
      ModDownOp::create(b, mup_bx, std_prod_bx, 25, Form::Eval, Form::Eval);
      ModDownOp::create(b, mup_ax, std_prod_ax, 25, Form::Eval, Form::Eval);

      // Stage 1b: Trace
      Value trace_bx = mup_bx;
      Value trace_ax = mup_ax;
      if (numSlots < halfDegree) {
        for (int64_t rot = numSlots; rot < halfDegree; rot <<= 1) {
          const auto& rotSpec =
              findEvkSpec(cheddar_ref::BootEvkKind::kRot, rot);
          auto rotEvk = viewEvk(rotSpec);
          Value next_bx = memref::AllocOp::create(b, l25Type);
          Value next_ax = memref::AllocOp::create(b, l25Type);
          HRotAddOp::create(b, next_bx, next_ax, trace_bx, trace_ax, trace_bx,
                            trace_ax, rotEvk.first, rotEvk.second, 25, rot,
                            /*galois=*/IntegerAttr{}, Form::Eval, Form::Eval);
          trace_bx = next_bx;
          trace_ax = next_ax;
        }
      }

      // Stage 2: CoeffToSlot
      Value cur_cts_bx = trace_bx;
      Value cur_cts_ax = trace_ax;
      for (size_t p = 0; p < ltPlan.cts.size(); ++p) {
        int64_t outLimbsP = 25 - p;
        auto phaseOutType = MemRefType::get({outLimbsP, degree}, intType);
        Value next_bx = memref::AllocOp::create(b, phaseOutType);
        Value next_ax = memref::AllocOp::create(b, phaseOutType);
        HoistPhaseOp::create(b, next_bx, next_ax, cur_cts_bx, cur_cts_ax,
                             func_consts_slab, func_evk_slab, "cts", p,
                             numSlots, paramsAttr);
        cur_cts_bx = next_bx;
        cur_cts_ax = next_ax;
      }

      // Stage 3: Extract + EvalMod
      const auto& conjSpec = findEvkSpec(cheddar_ref::BootEvkKind::kConj);
      auto conjEvk = viewEvk(conjSpec);
      const auto& relinSpec = findEvkSpec(cheddar_ref::BootEvkKind::kRelin);

      auto l21Type = MemRefType::get({22, degree}, intType);
      auto l13Type = MemRefType::get({14, degree}, intType);
      Value stc_in_bx, stc_in_ax;

      if (fullSlot) {
        Value conj_bx = memref::AllocOp::create(b, l21Type);
        Value conj_ax = memref::AllocOp::create(b, l21Type);
        HConjOp::create(b, conj_bx, conj_ax, cur_cts_bx, cur_cts_ax,
                        conjEvk.first, conjEvk.second, 21, Form::Eval,
                        Form::Eval);

        Value re_in_bx = memref::AllocOp::create(b, l21Type);
        Value re_in_ax = memref::AllocOp::create(b, l21Type);
        Add2Op::create(b, re_in_bx, re_in_ax, cur_cts_bx, cur_cts_ax, conj_bx,
                       conj_ax, Form::Eval, Form::Eval, Form::Eval,
                       DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{}, IntegerAttr{});

        Value im_diff_bx = memref::AllocOp::create(b, l21Type);
        Value im_diff_ax = memref::AllocOp::create(b, l21Type);
        // CRITICAL: ct_conj - main_ct (LHS is conj, RHS is cts)
        Sub2Op::create(b, im_diff_bx, im_diff_ax, conj_bx, conj_ax, cur_cts_bx,
                       cur_cts_ax, Form::Eval, Form::Eval, Form::Eval,
                       DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{}, IntegerAttr{});

        auto imagConsts = boot_consts.GetImaginaryUnitConsts();
        Value imag22 = getOrCreate1DGlobal(
            b, module, "__cheddar_imag_unit_22",
            ArrayRef<uint64_t>(imagConsts.data(), 22), intType);

        Value im_in_bx = memref::AllocOp::create(b, l21Type);
        Value im_in_ax = memref::AllocOp::create(b, l21Type);
        MulImaginaryUnitOp::create(b, im_in_bx, im_diff_bx, imag22, Form::Eval,
                                   Form::Eval, DenseI64ArrayAttr{},
                                   DenseI64ArrayAttr{});
        MulImaginaryUnitOp::create(b, im_in_ax, im_diff_ax, imag22, Form::Eval,
                                   Form::Eval, DenseI64ArrayAttr{},
                                   DenseI64ArrayAttr{});

        Value re_out_bx = memref::AllocOp::create(b, l13Type);
        Value re_out_ax = memref::AllocOp::create(b, l13Type);
        EvalModOp::create(b, re_out_bx, re_out_ax, re_in_bx, re_in_ax,
                          func_evk_slab, relinSpec.slab_byte_offset,
                          relinSpec.num_q, relinSpec.beta, Form::Eval,
                          Form::Eval, paramsAttr);

        Value im_eval_bx = memref::AllocOp::create(b, l13Type);
        Value im_eval_ax = memref::AllocOp::create(b, l13Type);
        EvalModOp::create(b, im_eval_bx, im_eval_ax, im_in_bx, im_in_ax,
                          func_evk_slab, relinSpec.slab_byte_offset,
                          relinSpec.num_q, relinSpec.beta, Form::Eval,
                          Form::Eval, paramsAttr);

        Value imag14 = getOrCreate1DGlobal(
            b, module, "__cheddar_imag_unit_14",
            ArrayRef<uint64_t>(imagConsts.data(), 14), intType);
        Value im_out_bx = memref::AllocOp::create(b, l13Type);
        Value im_out_ax = memref::AllocOp::create(b, l13Type);
        MulImaginaryUnitOp::create(b, im_out_bx, im_eval_bx, imag14, Form::Eval,
                                   Form::Eval, DenseI64ArrayAttr{},
                                   DenseI64ArrayAttr{});
        MulImaginaryUnitOp::create(b, im_out_ax, im_eval_ax, imag14, Form::Eval,
                                   Form::Eval, DenseI64ArrayAttr{},
                                   DenseI64ArrayAttr{});

        stc_in_bx = memref::AllocOp::create(b, l13Type);
        stc_in_ax = memref::AllocOp::create(b, l13Type);
        Add2Op::create(b, stc_in_bx, stc_in_ax, re_out_bx, re_out_ax, im_out_bx,
                       im_out_ax, Form::Eval, Form::Eval, Form::Eval,
                       DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                       DenseI64ArrayAttr{}, IntegerAttr{});
      } else {
        Value tmp_bx = memref::AllocOp::create(b, l21Type);
        Value tmp_ax = memref::AllocOp::create(b, l21Type);
        KeySwitchOp::create(b, tmp_bx, tmp_ax, cur_cts_bx, cur_cts_ax,
                            conjEvk.first, conjEvk.second, 21, Form::Eval,
                            Form::Eval);

        Value eval_in_bx = memref::AllocOp::create(b, l21Type);
        Value eval_in_ax = memref::AllocOp::create(b, l21Type);
        PermuteAccum2Op::create(
            b, eval_in_bx, eval_in_ax, cur_cts_bx, cur_cts_ax,
            ValueRange{tmp_bx}, ValueRange{tmp_ax}, Form::Eval,
            /*in_forms=*/ArrayAttr{}, b.getDenseI64ArrayAttr({2 * degree - 1}),
            /*out_limb_range=*/DenseI64ArrayAttr{},
            /*prime_offset=*/IntegerAttr{},
            /*num_aux=*/IntegerAttr{});

        stc_in_bx = memref::AllocOp::create(b, l13Type);
        stc_in_ax = memref::AllocOp::create(b, l13Type);
        EvalModOp::create(b, stc_in_bx, stc_in_ax, eval_in_bx, eval_in_ax,
                          func_evk_slab, relinSpec.slab_byte_offset,
                          relinSpec.num_q, relinSpec.beta, Form::Eval,
                          Form::Eval, paramsAttr);
      }

      // Stage 4: SlotToCoeff
      Value cur_stc_bx = stc_in_bx;
      Value cur_stc_ax = stc_in_ax;
      size_t numStcPhases = ltPlan.stc.size();
      for (size_t p = 0; p < numStcPhases; ++p) {
        int64_t outLimbsP = 13 - p;
        bool isLast = (p == numStcPhases - 1);
        if (isLast && fullSlot) {
          HoistPhaseOp::create(b, func_out_bx, func_out_ax, cur_stc_bx,
                               cur_stc_ax, func_consts_slab, func_evk_slab,
                               "stc", p, numSlots, paramsAttr);
        } else {
          auto phaseOutType = MemRefType::get({outLimbsP, degree}, intType);
          Value next_bx = memref::AllocOp::create(b, phaseOutType);
          Value next_ax = memref::AllocOp::create(b, phaseOutType);
          HoistPhaseOp::create(b, next_bx, next_ax, cur_stc_bx, cur_stc_ax,
                               func_consts_slab, func_evk_slab, "stc", p,
                               numSlots, paramsAttr);
          cur_stc_bx = next_bx;
          cur_stc_ax = next_ax;
        }
      }

      if (!fullSlot) {
        const auto& rotSpec =
            findEvkSpec(cheddar_ref::BootEvkKind::kRot, numSlots);
        auto rotSlotsEvk = viewEvk(rotSpec);
        HRotAddOp::create(b, func_out_bx, func_out_ax, cur_stc_bx, cur_stc_ax,
                          cur_stc_bx, cur_stc_ax, rotSlotsEvk.first,
                          rotSlotsEvk.second, 10, numSlots,
                          /*galois=*/IntegerAttr{}, Form::Eval, Form::Eval);
      }

      func::ReturnOp::create(b);
    }

    ImplicitLocOpBuilder b(loc, rewriter);

    // Pre-Boot LevelDown
    Value l0_bx, l0_ax;
    if (inLevel > 0) {
      auto l0Type = MemRefType::get({1, degree}, intType);
      l0_bx = memref::AllocOp::create(b, l0Type);
      l0_ax = memref::AllocOp::create(b, l0Type);
      emitLevelDown(b, module, boot_consts, op.getInBx(), op.getInAx(), inLevel,
                    0, l0_bx, l0_ax, degree, intType);
    } else {
      l0_bx = op.getInBx();
      l0_ax = op.getInAx();
    }

    // Call @__cheddar_boot_s{num_slots} + Post-Boot LevelDown
    if (outLevel == 10) {
      func::CallOp::create(
          b, bootFuncOp,
          ValueRange{op.getOutBx(), op.getOutAx(), l0_bx, l0_ax,
                     op.getConstsSlab(), op.getEvkSlab()});
    } else {
      auto l10Type = MemRefType::get({11, degree}, intType);
      Value l10_bx = memref::AllocOp::create(b, l10Type);
      Value l10_ax = memref::AllocOp::create(b, l10Type);
      func::CallOp::create(b, bootFuncOp,
                           ValueRange{l10_bx, l10_ax, l0_bx, l0_ax,
                                      op.getConstsSlab(), op.getEvkSlab()});
      emitLevelDown(b, module, boot_consts, l10_bx, l10_ax, 10, outLevel,
                    op.getOutBx(), op.getOutAx(), degree, intType);
    }

    rewriter.eraseOp(op);
    return success();
  }

 private:
  BootConstsCache& cache_;
};

struct ExpandBootstrapPass
    : public impl::ExpandBootstrapBase<ExpandBootstrapPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();

    BootConstsCache cache;
    RewritePatternSet patterns(context);
    patterns.add<ExpandBootPattern, ExpandEvalModPattern>(context, cache);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExpandBootstrapPass() {
  return std::make_unique<ExpandBootstrapPass>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
