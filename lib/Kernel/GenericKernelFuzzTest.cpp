#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
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
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"   // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"         // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

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

// Builds the kernel DAG for `op`: secret inputs packed in their layouts,
// cleartext inputs as their flat data.
FailureOr<NodePtr> buildKernel(const Op& op, std::string& reason) {
  GenericKernelSpec spec;
  spec.loopRanges = op.loopRanges;
  spec.numSlots = op.numSlots;
  std::vector<NodePtr> inputs;
  for (const Operand& input : op.inputs) {
    spec.inputs.push_back({input.map, input.layout});
    if (input.cleartext) {
      spec.inputs.back().cleartextShape = llvm::to_vector(input.tensor.shape);
      inputs.push_back(NodeTy::leaf(LiteralValue(input.tensor.data)));
    } else {
      inputs.push_back(NodeTy::leaf(pack(input, op.numSlots)));
    }
  }
  spec.output = {op.output.map, op.output.layout};
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
  // reversed, split into ciphertexts of k slots, replicated with a period, or
  // replicated with two copies per period at an irregular distance.
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
    // The smallest power of two of at least `n`.
    auto ceilPowerOfTwo = [](int64_t n) {
      int64_t p = 1;
      while (p < n) p *= 2;
      return p;
    };
    int64_t kind = pick(5);
    if (kind == 4 && ceilPowerOfTwo(2 * size + 1) > numSlots) kind = 3;
    switch (kind) {
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
      case 3: {
        int64_t period = ceilPowerOfTwo(size);
        placement = "ct = 0 and (slot " + linear(0, -1).substr(1) + ") mod " +
                    std::to_string(period) + " = 0";
        break;
      }
      default: {
        // Copies at flat and flat + distance in every period.
        int64_t period = ceilPowerOfTwo(2 * size + 1);
        int64_t distance = size + pick(period - 2 * size + 1);
        placement = "ct = 0 and exists (q, b : slot = " + linear(0, 1) + " + " +
                    std::to_string(period) + "q + " + std::to_string(distance) +
                    "b and 0 <= b <= 1)";
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

 public:
  // A strided window op: out[u, p] (+)= in[c, s * p + r] (* filter[u, c, r])
  // over output-only dims u, paired dims p with strides s, and reductions c
  // (read by the input only) and r (the window).
  Op buildStrided() {
    Op op;
    // Every operand fits in one ciphertext, as the layout family requires.
    op.numSlots = 512;
    op.kind = GenericReductionKind::Add;
    int64_t numPaired = 1 + pick(2);
    bool outputOnly = coin(), channel = coin();
    // Loops: [u], p..., [c], r...
    int64_t numLoops = (outputOnly ? 1 : 0) + 2 * numPaired + (channel ? 1 : 0);
    op.loopRanges.assign(numLoops, 1);
    int64_t loop = 0;
    SmallVector<AffineExpr> outputExprs, inputExprs, filterExprs;
    std::vector<int64_t> outputShape, inputShape, filterShape;
    if (outputOnly) {
      op.loopRanges[loop] = 1 + pick(3);
      outputExprs.push_back(getAffineDimExpr(loop, context));
      outputShape.push_back(op.loopRanges[loop]);
      filterExprs.push_back(getAffineDimExpr(loop, context));
      filterShape.push_back(op.loopRanges[loop]);
      ++loop;
    }
    int64_t firstPaired = loop;
    loop += numPaired;
    if (channel) {
      op.loopRanges[loop] = 1 + pick(3);
      inputExprs.push_back(getAffineDimExpr(loop, context));
      inputShape.push_back(op.loopRanges[loop]);
      filterExprs.push_back(getAffineDimExpr(loop, context));
      filterShape.push_back(op.loopRanges[loop]);
      ++loop;
    }
    for (int64_t i = 0; i < numPaired; ++i) {
      int64_t p = firstPaired + i, r = loop + i;
      int64_t stride = 1 + pick(3), window = 1 + pick(3);
      op.loopRanges[p] = 1 + pick(4);
      op.loopRanges[r] = window;
      outputExprs.push_back(getAffineDimExpr(p, context));
      outputShape.push_back(op.loopRanges[p]);
      inputExprs.push_back(stride * getAffineDimExpr(p, context) +
                           getAffineDimExpr(r, context));
      inputShape.push_back(stride * (op.loopRanges[p] - 1) + window);
      filterExprs.push_back(getAffineDimExpr(r, context));
      filterShape.push_back(window);
    }
    op.inputs.push_back(
        makeOperand(AffineMap::get(numLoops, 0, inputExprs, context),
                    inputShape, randomTensor(inputShape, 4, false), op.numSlots,
                    /*allowPadding=*/true));
    if (coin()) {
      Operand filter;
      filter.map = AffineMap::get(numLoops, 0, filterExprs, context);
      filter.tensor = randomTensor(filterShape, 4, false);
      filter.cleartext = true;
      op.inputs.push_back(std::move(filter));
      op.body = Body::Product;
    } else {
      op.body = Body::First;
    }
    op.initKind = GenericInitKind::Neutral;
    op.output = makeOperand(AffineMap::get(numLoops, 0, outputExprs, context),
                            outputShape, Tensor(outputShape), op.numSlots,
                            /*allowPadding=*/false);
    return op;
  }

 private:
  std::mt19937_64 rng;
  MLIRContext* context;
};

// The number of ciphertexts a layout uses.
int64_t numCiphertexts(const IntegerRelation& layout) {
  PointPairCollector collector(layout.getNumDomainVars(), 2);
  enumeratePoints(layout, collector);
  int64_t numCts = 1;
  for (const auto& [index, position] : collector.points)
    numCts = std::max(numCts, position[0] + 1);
  return numCts;
}

// The polyphase layout of the secret input of `op`, or the input's own
// layout if it is aligned or no polyphase layout applies.
std::optional<IntegerRelation> polyphaseLayout(const Op& op,
                                               std::string& reason) {
  const Operand& input = op.inputs.front();
  FailureOr<std::optional<IntegerRelation>> layout = getPolyphaseInputLayout(
      op.loopRanges, input.map, input.layout, op.output.map, op.output.layout,
      op.output.tensor.shape, op.numSlots, reason);
  if (failed(layout)) return std::nullopt;
  return *layout;
}

AffineMap parseMap(StringRef str, MLIRContext* context) {
  return cast<AffineMapAttr>(parseAttribute(str, context)).getValue();
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

// The groups the kernel uses for `op`.
InstanceGroups getGroups(const Op& op) {
  std::string reason;
  FailureOr<InstanceGroups> groups =
      groupInstancesByCost(getSpec(op), op.output.numCts, op.kind, reason);
  EXPECT_TRUE(succeeded(groups)) << reason;
  return *groups;
}

int64_t numGroups(const Op& op) {
  return getGroupCost(getGroups(op), op.numSlots).groups;
}

// A strided window op computes the same result with its input packed in the
// polyphase layout, under which the input is aligned.
void polyphaseLayoutMatchesNaive(uint64_t seed) {
  MLIRContext context;
  Op op = RandomOpBuilder(seed, &context).buildStrided();
  std::string reason;
  std::optional<IntegerRelation> polyphase = polyphaseLayout(op, reason);
  if (!polyphase) return;
  Operand& input = op.inputs.front();
  input.layout = *polyphase;
  input.numCts = numCiphertexts(*polyphase);
  FailureOr<std::optional<IntegerRelation>> again = getPolyphaseInputLayout(
      op.loopRanges, input.map, input.layout, op.output.map, op.output.layout,
      op.output.tensor.shape, op.numSlots, reason);
  ASSERT_TRUE(succeeded(again)) << reason;
  EXPECT_FALSE(again->has_value()) << "the polyphase layout is not aligned";
  FailureOr<NodePtr> dag = buildKernel(op, reason);
  ASSERT_TRUE(succeeded(dag)) << reason;
  checkKernel(op, *dag);
}

FUZZ_TEST(GenericKernelFuzzTest, polyphaseLayoutMatchesNaive)
    .WithDomains(fuzztest::Arbitrary<uint64_t>());

// A stride-2 3x3 convolution over a row-major input with "same" padding (the
// halo is unplaced): the polyphase layout needs a few groups per filter tap
// instead of a few per output position.
TEST(GenericKernelTest, PolyphaseLayoutOfStridedConvolution) {
  MLIRContext context;
  auto parse = [&](StringRef str) { return parseMap(str, &context); };
  Op op;
  op.numSlots = 64;
  op.kind = GenericReductionKind::Add;
  op.body = Body::Product;
  op.initKind = GenericInitKind::Neutral;
  // (co, y, x, ci, ky, kx)
  op.loopRanges = {2, 4, 4, 2, 3, 3};
  Operand input;
  input.map = parse(
      "affine_map<(co, y, x, ci, ky, kx) -> (ci, 2 * y + ky, 2 * x + kx)>");
  input.tensor = Tensor::iota({2, 10, 10});
  for (auto [flat, v] : llvm::enumerate(input.tensor.data)) {
    int64_t row = flat / 10 % 10, col = flat % 10;
    if (row == 0 || row == 9 || col == 0 || col == 9) v = 0;
  }
  input.layout = *getIntegerRelationFromIslStr(
      "{ [c, y, x] -> [ct, slot] : 64ct + slot = 64c + 8y + x - 9 and "
      "0 <= slot <= 63 and 0 <= c <= 1 and 1 <= y <= 8 and 1 <= x <= 8 }");
  input.numCts = 2;
  op.inputs.push_back(input);
  Operand filter;
  filter.map = parse("affine_map<(co, y, x, ci, ky, kx) -> (co, ci, ky, kx)>");
  filter.tensor = Tensor::iota({2, 2, 3, 3});
  filter.cleartext = true;
  op.inputs.push_back(filter);
  op.output.map = parse("affine_map<(co, y, x, ci, ky, kx) -> (co, y, x)>");
  op.output.tensor = Tensor({2, 4, 4});
  op.output.layout = *getIntegerRelationFromIslStr(
      "{ [c, y, x] -> [ct, slot] : ct = 0 and slot = 16c + 4y + x and "
      "0 <= c <= 1 and 0 <= y <= 3 and 0 <= x <= 3 }");

  int64_t before = numGroups(op);
  std::string reason;
  std::optional<IntegerRelation> polyphase = polyphaseLayout(op, reason);
  ASSERT_TRUE(polyphase.has_value()) << reason;
  op.inputs.front().layout = *polyphase;
  op.inputs.front().numCts = numCiphertexts(*polyphase);
  EXPECT_LE(op.inputs.front().numCts, 2);
  int64_t after = numGroups(op);
  // Two output channels times two input channels times nine taps, split at
  // most once more where a slab of the input crosses a ciphertext.
  EXPECT_LE(after, 2 * 2 * 9 * 2);
  EXPECT_LT(after, before);
  FailureOr<NodePtr> dag = buildKernel(op, reason);
  ASSERT_TRUE(succeeded(dag)) << reason;
  checkKernel(op, *dag);
}

// A squeeze-excite style reduction of a 4x16x16 input into 4 channels, out[co]
// = sum W[co, ci] * x[ci, h, w], with the result repeated every 4 slots. One
// output copy per class of slots mod 4 needs 4 groups (the shift of x mod 4)
// and a rotate-and-reduce over the 256 copies; one group per output copy
// would need 1024 rotations.
TEST(GenericKernelTest, ReductionIntoReplicatedResult) {
  MLIRContext context;
  Op op;
  op.numSlots = 1024;
  op.kind = GenericReductionKind::Add;
  op.body = Body::Product;
  op.initKind = GenericInitKind::Neutral;
  op.loopRanges = {4, 4, 16, 16};
  Operand input;
  input.map = parseMap("affine_map<(co, ci, h, w) -> (ci, h, w)>", &context);
  input.tensor = Tensor::iota({4, 16, 16});
  input.layout = *getIntegerRelationFromIslStr(
      "{ [c, h, w] -> [ct, slot] : ct = 0 and slot = 256c + 16h + w and "
      "0 <= c <= 3 and 0 <= h <= 15 and 0 <= w <= 15 }");
  op.inputs.push_back(input);
  Operand weights;
  weights.map = parseMap("affine_map<(co, ci, h, w) -> (co, ci)>", &context);
  weights.tensor = Tensor::iota({4, 4});
  weights.cleartext = true;
  op.inputs.push_back(weights);
  op.output.map = parseMap("affine_map<(co, ci, h, w) -> (co)>", &context);
  op.output.tensor = Tensor({4});
  op.output.layout = *getIntegerRelationFromIslStr(
      "{ [c] -> [ct, slot] : ct = 0 and (slot - c) mod 4 = 0 and "
      "0 <= c <= 3 and 0 <= slot <= 1023 }");

  EXPECT_EQ(getSlotPeriod(op.output.layout, op.numSlots), 4);
  InstanceGroups groups = getGroups(op);
  EXPECT_EQ(groups.period, 4);
  GroupCost cost = getGroupCost(groups, op.numSlots);
  EXPECT_EQ(cost.groups, 4);
  // Three shifts of x and log2(256) rotations of the rotate-and-reduce.
  EXPECT_EQ(cost.rotations, 3 + 8);
  std::string reason;
  FailureOr<NodePtr> dag = buildKernel(op, reason);
  ASSERT_TRUE(succeeded(dag)) << reason;
  checkKernel(op, *dag);
}

// An elementwise op on operands repeated like the result needs no rotation
// with one group per output copy, so the kernel keeps that grouping.
TEST(GenericKernelTest, ElementwiseOnReplicatedOperands) {
  MLIRContext context;
  Op op;
  op.numSlots = 64;
  op.kind = GenericReductionKind::None;
  op.body = Body::Product;
  op.initKind = GenericInitKind::Empty;
  op.loopRanges = {4};
  std::string layout =
      "{ [c] -> [ct, slot] : ct = 0 and (slot - c) mod 4 = 0 and "
      "0 <= c <= 3 and 0 <= slot <= 63 }";
  for (int start : {1, 5}) {
    Operand input;
    input.map = parseMap("affine_map<(c) -> (c)>", &context);
    input.tensor = Tensor::iota({4}, start);
    input.layout = *getIntegerRelationFromIslStr(layout);
    op.inputs.push_back(input);
  }
  op.output.map = parseMap("affine_map<(c) -> (c)>", &context);
  op.output.tensor = Tensor({4});
  op.output.layout = *getIntegerRelationFromIslStr(layout);

  InstanceGroups groups = getGroups(op);
  EXPECT_EQ(groups.period, op.numSlots);
  GroupCost cost = getGroupCost(groups, op.numSlots);
  EXPECT_EQ(cost.groups, 1);
  EXPECT_EQ(cost.rotations, 0);
  std::string reason;
  FailureOr<NodePtr> dag = buildKernel(op, reason);
  ASSERT_TRUE(succeeded(dag)) << reason;
  checkKernel(op, *dag);
}

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

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
