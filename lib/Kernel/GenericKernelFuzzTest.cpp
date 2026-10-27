#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/GenericKernelImplementation.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/GenericKernel.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"        // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

// Checks the generic structured-op kernel end to end on literal data: pack the
// operands with their layouts, evaluate the kernel's arithmetic DAG, unpack
// the result, and compare with a scalar loop nest evaluating the op.
//
// EvalVisitor does not evaluate opaque elementwise ops, so the bodies here are
// built from add and mul, and idempotent reductions use AND (a product of 0/1
// values). Bodies with comparisons, selects or casts are covered by the lit
// tests of convert-to-ciphertext-semantics.
namespace mlir {
namespace heir {
namespace kernel {
namespace {

using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using NodeTy = ArithmeticDagNode<LiteralValue>;
using NodePtr = std::shared_ptr<NodeTy>;

// A dense row-major tensor.
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<int> data;

  Tensor() = default;
  explicit Tensor(std::vector<int64_t> shape, int fill = 0) : shape(shape) {
    int64_t size = 1;
    for (int64_t d : shape) size *= d;
    data.assign(size, fill);
  }

  static Tensor iota(std::vector<int64_t> shape, int start = 1) {
    Tensor t(std::move(shape));
    std::iota(t.data.begin(), t.data.end(), start);
    return t;
  }

  int64_t stride(int64_t dim) const {
    int64_t result = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(shape.size()); ++d)
      result *= shape[d];
    return result;
  }
  int64_t flatten(ArrayRef<int64_t> index) const {
    int64_t flat = 0;
    for (auto [i, v] : llvm::enumerate(index)) flat = flat * shape[i] + v;
    return flat;
  }
  int& at(ArrayRef<int64_t> index) { return data[flatten(index)]; }
  int at(ArrayRef<int64_t> index) const { return data[flatten(index)]; }
};

// The slot-wise computation of a group, with a scalar reference.
enum class Body {
  First,             // a
  Product,           // a * b
  Sum,               // a + b
  Difference,        // a - b
  Scaled,            // 3 * a
  ProductPlusFirst,  // a * b + a
  PlusInit,          // out + a (reads the init; no reductions)
  kNumBodies,
};

int evalBody(Body body, ArrayRef<int> in, int out) {
  int a = in.front(), b = in.back();
  switch (body) {
    case Body::First:
      return a;
    case Body::Product:
      return a * b;
    case Body::Sum:
      return a + b;
    case Body::Difference:
      return a - b;
    case Body::Scaled:
      return 3 * a;
    case Body::ProductPlusFirst:
      return a * b + a;
    case Body::PlusInit:
      return out + a;
    case Body::kNumBodies:
      break;
  }
  return 0;
}

GenericBodyBuilder<LiteralValue> bodyBuilder(Body body, int64_t numSlots) {
  return [=](const std::vector<NodePtr>& in, const NodePtr& init) {
    NodePtr a = in.front(), b = in.back();
    switch (body) {
      case Body::First:
        return a;
      case Body::Product:
        return NodeTy::mul(a, b);
      case Body::Sum:
        return NodeTy::add(a, b);
      case Body::Difference:
        return NodeTy::sub(a, b);
      case Body::Scaled:
        return NodeTy::mul(
            NodeTy::splat(3, DagType::intTensor(32, {1, numSlots})), a);
      case Body::ProductPlusFirst:
        return NodeTy::add(NodeTy::mul(a, b), a);
      case Body::PlusInit:
        return NodeTy::add(init, a);
      case Body::kNumBodies:
        break;
    }
    return a;
  };
}

SmallVector<int64_t> apply(AffineMap map, ArrayRef<int64_t> point) {
  SmallVector<int64_t> result;
  SmallVector<AffineExpr> dims;
  for (int64_t v : point)
    dims.push_back(getAffineConstantExpr(v, map.getContext()));
  for (AffineExpr expr : map.getResults()) {
    result.push_back(
        cast<AffineConstantExpr>(expr.replaceDims(dims)).getValue());
  }
  return result;
}

struct Operand {
  AffineMap map;
  IntegerRelation layout{PresburgerSpace::getRelationSpace(0, 2)};
  int64_t numCts = 1;
  Tensor tensor;
  // Cleartext operands have no layout.
  bool cleartext = false;
};

struct Op {
  SmallVector<int64_t> loopRanges;
  std::vector<Operand> inputs;
  // The tensor of `output` is the init.
  Operand output;
  GenericReductionKind kind = GenericReductionKind::None;
  GenericInitKind initKind = GenericInitKind::Neutral;
  Body body = Body::First;
  int64_t numSlots = 0;
};

// Evaluates the op with scalar loops.
Tensor evalNaive(const Op& op) {
  Tensor out = op.output.tensor;
  SmallVector<int64_t> d(op.loopRanges.size(), 0);
  while (true) {
    std::vector<int> values;
    for (const Operand& input : op.inputs) {
      values.push_back(input.tensor.at(apply(input.map, d)));
    }
    int& acc = out.at(apply(op.output.map, d));
    int y = evalBody(op.body, values, acc);
    switch (op.kind) {
      case GenericReductionKind::None:
        acc = y;
        break;
      case GenericReductionKind::Add:
        acc += y;
        break;
      case GenericReductionKind::Mul:
        acc *= y;
        break;
      case GenericReductionKind::Idempotent:
        acc = acc && y;
        break;
    }
    int64_t j = static_cast<int64_t>(d.size()) - 1;
    while (j >= 0 && ++d[j] == op.loopRanges[j]) {
      d[j] = 0;
      --j;
    }
    if (j < 0) break;
  }
  return out;
}

LiteralValue pack(const Operand& operand, int64_t numSlots) {
  std::function<int(const std::vector<int64_t>&)> get =
      [&](const std::vector<int64_t>& index) {
        return operand.tensor.at(index);
      };
  return LiteralValue(evaluateLayout<int>(
      operand.layout, get, SmallVector<int64_t>{operand.numCts, numSlots}));
}

GenericKernelSpec getSpec(const Op& op) {
  GenericKernelSpec spec;
  spec.loopRanges = op.loopRanges;
  spec.numSlots = op.numSlots;
  for (const Operand& input : op.inputs) {
    spec.inputs.push_back({input.map, input.layout});
    if (input.cleartext)
      spec.inputs.back().cleartextShape = llvm::to_vector(input.tensor.shape);
  }
  spec.output = {op.output.map, op.output.layout};
  return spec;
}

// Builds the kernel DAG for `op`: secret inputs packed in their layouts,
// cleartext inputs as their flat data.
FailureOr<NodePtr> buildKernel(const Op& op, std::string& reason) {
  GenericKernelSpec spec = getSpec(op);
  std::vector<NodePtr> inputs;
  for (const Operand& input : op.inputs) {
    inputs.push_back(input.cleartext
                         ? NodeTy::leaf(LiteralValue(input.tensor.data))
                         : NodeTy::leaf(pack(input, op.numSlots)));
  }
  GenericKernelConfig config;
  config.kind = op.kind;
  config.initKind = op.initKind;
  config.bodyReadsInit = op.body == Body::PlusInit;
  config.numCts = op.output.numCts;
  config.elementType = DagType::integer(32);
  config.inputElementTypes.assign(op.inputs.size(), DagType::integer(32));
  // Idempotent reductions are tested with AND on 0/1 values.
  return implementGenericKernel<LiteralValue>(
      spec, inputs, NodeTy::leaf(pack(op.output, op.numSlots)), config,
      bodyBuilder(op.body, op.numSlots),
      [](const NodePtr& acc, const NodePtr& term) {
        return NodeTy::mul(acc, term);
      },
      reason);
}

// Checks every slot of every element of the result layout against the naive
// evaluation.
void checkKernel(const Op& op, const NodePtr& dag) {
  auto packed =
      std::get<std::vector<std::vector<int>>>(evalKernel(dag)[0].get());
  ASSERT_EQ(packed.size(), static_cast<size_t>(op.output.numCts));

  Tensor expected = evalNaive(op);
  unpackLayout<int>(op.output.layout, packed,
                    [&](const std::vector<int64_t>& index, int value) {
                      std::string where;
                      for (int64_t v : index) where += std::to_string(v) + " ";
                      EXPECT_EQ(value, expected.at(index))
                          << "at data index [ " << where << "]";
                    });
}

// Random ops from a family of loop nests, indexing maps and layouts.
class RandomOpBuilder {
 public:
  RandomOpBuilder(uint64_t seed, MLIRContext* context)
      : rng(seed), context(context) {}

  Op build() {
    Op op;
    op.numSlots = 128;
    int64_t rank = 1 + pick(3);
    for (int64_t j = 0; j < rank; ++j) op.loopRanges.push_back(1 + pick(4));
    op.kind = static_cast<GenericReductionKind>(pick(4));

    // Reductions get at least one reduction loop.
    std::vector<bool> isReduction(rank, false);
    if (op.kind != GenericReductionKind::None) {
      for (int64_t j = 0; j < rank; ++j) isReduction[j] = coin();
      isReduction[pick(rank)] = true;
    }
    SmallVector<AffineExpr> outputExprs;
    std::vector<int64_t> outputShape;
    for (int64_t j = 0; j < rank; ++j) {
      if (isReduction[j]) continue;
      outputExprs.push_back(getAffineDimExpr(j, context));
      outputShape.push_back(op.loopRanges[j]);
    }
    // Ops without reductions and idempotent reductions may write a window of
    // a larger result.
    bool partial = (op.kind == GenericReductionKind::None ||
                    op.kind == GenericReductionKind::Idempotent) &&
                   !outputExprs.empty() && coin();
    if (partial) {
      int64_t offset = pick(3);
      outputExprs[0] = outputExprs[0] + offset;
      outputShape[0] += offset + pick(3);
    }

    // Small values keep products in range; AND (idempotent) needs 0/1.
    int magnitude = op.kind == GenericReductionKind::Mul ? 1 : 4;
    bool binary = op.kind == GenericReductionKind::Idempotent;
    int64_t numInputs = 1 + pick(2);
    for (int64_t i = 0; i < numInputs; ++i) {
      std::vector<int64_t> dims(rank);
      std::iota(dims.begin(), dims.end(), 0);
      std::shuffle(dims.begin(), dims.end(), rng);
      dims.resize(1 + pick(std::min<int64_t>(rank, 2)));
      SmallVector<AffineExpr> exprs;
      std::vector<int64_t> shape;
      for (int64_t dim : dims) {
        AffineExpr expr = getAffineDimExpr(dim, context);
        int64_t extent = op.loopRanges[dim];
        // A sliding window over a second loop.
        int64_t other = pick(rank);
        if (other != dim && pick(4) == 0) {
          expr = expr + getAffineDimExpr(other, context);
          extent += op.loopRanges[other] - 1;
        }
        // A reversed (flipped) dimension.
        if (pick(4) == 0) expr = (extent - 1) - expr;
        exprs.push_back(expr);
        shape.push_back(extent);
      }
      AffineMap map = AffineMap::get(rank, 0, exprs, context);
      Tensor tensor = randomTensor(shape, magnitude, binary);
      if (pick(3) == 0) {
        Operand cleartext;
        cleartext.map = map;
        cleartext.tensor = std::move(tensor);
        cleartext.cleartext = true;
        op.inputs.push_back(std::move(cleartext));
        continue;
      }
      op.inputs.push_back(makeOperand(map, shape, std::move(tensor),
                                      op.numSlots, /*allowPadding=*/true));
    }

    std::vector<Body> bodies = {Body::First,  Body::Product,
                                Body::Sum,    Body::Difference,
                                Body::Scaled, Body::ProductPlusFirst};
    if (op.kind == GenericReductionKind::None) bodies.push_back(Body::PlusInit);
    if (op.kind == GenericReductionKind::Mul ||
        op.kind == GenericReductionKind::Idempotent)
      bodies = {Body::First, Body::Product};
    op.body = bodies[pick(bodies.size())];

    // The init: undefined (only for full writes that do not read it, or for
    // full idempotent reductions, which never read their identity), the
    // identity, or arbitrary.
    std::vector<GenericInitKind> initKinds = {GenericInitKind::Neutral,
                                              GenericInitKind::Other};
    if ((op.kind == GenericReductionKind::None && !partial &&
         op.body != Body::PlusInit) ||
        (op.kind == GenericReductionKind::Idempotent && !partial)) {
      initKinds.push_back(GenericInitKind::Empty);
    }
    // The kernel rejects partial idempotent reductions into their identity.
    if (op.kind == GenericReductionKind::Idempotent && partial)
      initKinds = {GenericInitKind::Other};
    op.initKind = initKinds[pick(initKinds.size())];
    Tensor init(outputShape, op.kind == GenericReductionKind::Mul ||
                                     op.kind == GenericReductionKind::Idempotent
                                 ? 1
                                 : 0);
    if (op.initKind == GenericInitKind::Other)
      init = randomTensor(outputShape, magnitude, binary);
    op.output = makeOperand(AffineMap::get(rank, 0, outputExprs, context),
                            outputShape, init, op.numSlots,
                            /*allowPadding=*/false);
    return op;
  }

 private:
  int64_t pick(int64_t n) {
    return std::uniform_int_distribution<int64_t>(0, n - 1)(rng);
  }
  bool coin() { return pick(2) == 1; }

  Tensor randomTensor(std::vector<int64_t> shape, int magnitude, bool binary) {
    Tensor t(std::move(shape));
    for (int& v : t.data) {
      v = binary ? static_cast<int>(pick(2))
                 : static_cast<int>(pick(2 * magnitude + 1)) - magnitude;
    }
    return t;
  }

  // A layout from the family: row-major order over a random permutation of
  // the dimensions, placed in one ciphertext at an offset and stride,
  // reversed, split into ciphertexts of k slots, or replicated with a period.
  // With `allowPadding`, the layout may omit a border of the tensor, which
  // then holds zeros (as zero padding does).
  Operand makeOperand(AffineMap map, ArrayRef<int64_t> shape, Tensor tensor,
                      int64_t numSlots, bool allowPadding) {
    int64_t rank = shape.size();
    int64_t size = 1;
    for (int64_t d : shape) size *= d;
    std::vector<int64_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    // The row-major flattening over the permuted dimensions, as coefficients.
    std::vector<int64_t> coefficients(rank, 0);
    int64_t stride = 1;
    for (int64_t p = rank - 1; p >= 0; --p) {
      coefficients[perm[p]] = stride;
      stride *= shape[perm[p]];
    }
    // constant + scale * flat, as an ISL expression.
    auto linear = [&](int64_t constant, int64_t scale) {
      std::string expr = std::to_string(constant);
      for (int64_t p = 0; p < rank; ++p) {
        int64_t c = scale * coefficients[p];
        expr += (c < 0 ? " - " : " + ") + std::to_string(std::abs(c)) + "i" +
                std::to_string(p);
      }
      return expr;
    };
    std::string vars, bounds = "0 <= slot <= " + std::to_string(numSlots - 1);
    for (int64_t p = 0; p < rank; ++p) {
      vars += (p ? ", i" : "i") + std::to_string(p);
      bounds += " and 0 <= i" + std::to_string(p) +
                " <= " + std::to_string(shape[p] - 1);
    }

    int64_t numCts = 1;
    std::string placement;
    switch (pick(4)) {
      case 0: {
        int64_t step = 2 * size <= numSlots && coin() ? 2 : 1;
        int64_t offset = pick(numSlots - step * (size - 1));
        placement = "ct = 0 and slot = " + linear(offset, step);
        break;
      }
      case 1:
        placement = "ct = 0 and slot = " + linear(size - 1, -1);
        break;
      case 2: {
        int64_t k = int64_t{1} << pick(8);
        numCts = (size + k - 1) / k;
        placement = std::to_string(k) + "ct + slot = " + linear(0, 1) +
                    " and 0 <= slot <= " + std::to_string(k - 1);
        break;
      }
      default: {
        int64_t period = 1;
        while (period < size) period *= 2;
        placement = "ct = 0 and (slot " + linear(0, -1).substr(1) + ") mod " +
                    std::to_string(period) + " = 0";
        break;
      }
    }
    if (allowPadding && pick(3) == 0) {
      for (int64_t p = 0; p < rank; ++p) {
        int64_t margin = shape[p] > 2 ? 1 : 0;
        int64_t low = pick(margin + 1), high = shape[p] - 1 - pick(margin + 1);
        bounds += " and " + std::to_string(low) + " <= i" + std::to_string(p) +
                  " <= " + std::to_string(high);
        for (auto [flat, v] : llvm::enumerate(tensor.data)) {
          int64_t index = flat / tensor.stride(p) % shape[p];
          if (index < low || index > high) v = 0;
        }
      }
    }
    std::string isl = "{ [" + vars + "] -> [ct, slot] : " + placement +
                      " and " + bounds + " }";
    FailureOr<IntegerRelation> layout = getIntegerRelationFromIslStr(isl);
    EXPECT_TRUE(succeeded(layout)) << isl;
    return {map, *layout, numCts, std::move(tensor)};
  }

  std::mt19937_64 rng;
  MLIRContext* context;
};

void genericKernelMatchesNaive(uint64_t seed) {
  MLIRContext context;
  Op op = RandomOpBuilder(seed, &context).build();
  std::string reason;
  FailureOr<NodePtr> dag = buildKernel(op, reason);
  // The only rejection the generator can hit: a repeated factor of a product.
  if (failed(dag)) {
    EXPECT_EQ(op.kind, GenericReductionKind::Mul) << reason;
    return;
  }
  checkKernel(op, *dag);
}

FUZZ_TEST(GenericKernelFuzzTest, genericKernelMatchesNaive)
    .WithDomains(fuzztest::Arbitrary<uint64_t>());

// The symbolic count of groups agrees with the walk whenever it stays within
// its ISL budget.
void groupCountMatchesWalk(uint64_t seed) {
  MLIRContext context;
  Op op = RandomOpBuilder(seed, &context).build();
  GenericKernelSpec spec = getSpec(op);
  std::string reason;
  FailureOr<InstanceGroups> groups =
      groupInstances(spec, op.output.numCts, GenericReductionKind::Add, reason);
  ASSERT_TRUE(succeeded(groups)) << reason;
  FailureOr<GroupCount> count = countInstanceGroups(
      spec, /*maxKeys=*/1 << 20, /*maxOperations=*/1000000, reason);
  if (failed(count)) {
    EXPECT_NE(reason.find("ISL ran out of operations"), std::string::npos)
        << reason;
    return;
  }
  EXPECT_EQ(count->numKeys, static_cast<int64_t>(groups->readsByKey.size()));
  // Distinct nonzero rotations per secret input.
  std::set<std::tuple<size_t, int64_t, int64_t>> rotations;
  for (const auto& [key, slots] : groups->readsByKey) {
    for (size_t k = 1; k + 1 < key.size(); k += 2) {
      if (key[k + 1] != 0) rotations.insert({k, key[k], key[k + 1]});
    }
  }
  EXPECT_EQ(count->numRotations, static_cast<int64_t>(rotations.size()));
}

FUZZ_TEST(GenericKernelFuzzTest, groupCountMatchesWalk)
    .WithDomains(fuzztest::Arbitrary<uint64_t>());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
