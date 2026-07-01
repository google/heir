#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Transforms/LinalgCanonicalizations/ConvPoolFusion.h"
#include "llvm/include/llvm/ADT/APFloat.h"  // from @llvm-project

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace {

std::vector<double> referenceConv2D(const std::vector<double>& input,
                                    const std::vector<double>& filter,
                                    const Conv2DSizeParams& conv, int64_t inH,
                                    int64_t inW, int64_t outH, int64_t outW) {
  int64_t F = conv.F;
  int64_t C = conv.C;
  int64_t kH = conv.kH;
  int64_t kW = conv.kW;
  int64_t strideH = conv.strideH;
  int64_t strideW = conv.strideW;
  int64_t dilationH = conv.dilationH;
  int64_t dilationW = conv.dilationW;

  std::vector<double> output(F * outH * outW, 0.0);

  for (int64_t f = 0; f < F; ++f) {
    for (int64_t oh = 0; oh < outH; ++oh) {
      for (int64_t ow = 0; ow < outW; ++ow) {
        double sum = 0.0;
        for (int64_t c = 0; c < C; ++c) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t ih = oh * strideH + kh * dilationH;
              int64_t iw = ow * strideW + kw * dilationW;

              int64_t inputIdx = (c * inH + ih) * inW + iw;
              int64_t filterIdx = ((f * C + c) * kH + kh) * kW + kw;

              sum += input[inputIdx] * filter[filterIdx];
            }
          }
        }
        output[(f * outH + oh) * outW + ow] = sum;
      }
    }
  }
  return output;
}

std::vector<double> referencePoolSum(const std::vector<double>& input,
                                     int64_t F, const Pool2DSizeParams& pool,
                                     int64_t inH, int64_t inW, int64_t outH,
                                     int64_t outW) {
  int64_t pH = pool.pH;
  int64_t pW = pool.pW;
  int64_t strideH = pool.strideH;
  int64_t strideW = pool.strideW;
  int64_t dilationH = pool.dilationH;
  int64_t dilationW = pool.dilationW;

  std::vector<double> output(F * outH * outW, 0.0);

  for (int64_t f = 0; f < F; ++f) {
    for (int64_t oh = 0; oh < outH; ++oh) {
      for (int64_t ow = 0; ow < outW; ++ow) {
        double sum = 0.0;
        for (int64_t ph = 0; ph < pH; ++ph) {
          for (int64_t pw = 0; pw < pW; ++pw) {
            int64_t ih = oh * strideH + ph * dilationH;
            int64_t iw = ow * strideW + pw * dilationW;

            int64_t inputIdx = (f * inH + ih) * inW + iw;
            sum += input[inputIdx];
          }
        }
        output[(f * outH + oh) * outW + ow] = sum;
      }
    }
  }
  return output;
}

TEST(ConvPoolFusionTest, KnownInputTest) {
  Conv2DSizeParams conv;
  conv.F = 1;
  conv.C = 1;
  conv.kH = 2;
  conv.kW = 2;
  conv.strideH = 2;
  conv.strideW = 2;
  conv.dilationH = 1;
  conv.dilationW = 1;

  Pool2DSizeParams pool;
  pool.pH = 2;
  pool.pW = 2;
  pool.strideH = 1;
  pool.strideW = 1;
  pool.dilationH = 1;
  pool.dilationW = 1;

  std::vector<llvm::APFloat> filter = {llvm::APFloat(1.0f), llvm::APFloat(2.0f),
                                       llvm::APFloat(3.0f),
                                       llvm::APFloat(4.0f)};

  auto fusedFilter = fuseFloatFilters(
      filter, conv, pool, llvm::APFloat::IEEEsingle(), std::nullopt);

  std::vector<float> expected = {1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                 3.0f, 4.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                                 3.0f, 4.0f, 3.0f, 4.0f};

  std::vector<float> fusedFilterFloats;
  fusedFilterFloats.reserve(fusedFilter.size());
  for (const auto& val : fusedFilter) {
    fusedFilterFloats.push_back(val.convertToFloat());
  }

  EXPECT_THAT(fusedFilterFloats,
              ::testing::Pointwise(::testing::FloatNear(1e-5), expected));
}

void FuzzConvPoolEquivalence(const Conv2DSizeParams& conv,
                             const Pool2DSizeParams& pool,
                             const OutputSizeParams& out,
                             const std::vector<double>& rawInput,
                             const std::vector<double>& rawFilter,
                             double scale) {
  int64_t poolOutH = out.poolOutH;
  int64_t poolOutW = out.poolOutW;

  // Compute required conv output size (pool input size)
  int64_t convOutH =
      (poolOutH - 1) * pool.strideH + (pool.pH - 1) * pool.dilationH + 1;
  int64_t convOutW =
      (poolOutW - 1) * pool.strideW + (pool.pW - 1) * pool.dilationW + 1;

  // Compute required conv input size
  int64_t inH =
      (convOutH - 1) * conv.strideH + (conv.kH - 1) * conv.dilationH + 1;
  int64_t inW =
      (convOutW - 1) * conv.strideW + (conv.kW - 1) * conv.dilationW + 1;

  // Resize input
  int64_t inputSize = conv.C * inH * inW;
  std::vector<double> input = rawInput;
  input.resize(inputSize, 0.0);

  // Resize filter
  int64_t filterSize = conv.F * conv.C * conv.kH * conv.kW;
  std::vector<double> filter = rawFilter;
  filter.resize(filterSize, 0.0);

  // Run cascade: Conv2D -> PoolSum -> Scale
  auto convOut =
      referenceConv2D(input, filter, conv, inH, inW, convOutH, convOutW);
  auto poolOut = referencePoolSum(convOut, conv.F, pool, convOutH, convOutW,
                                  poolOutH, poolOutW);

  std::vector<double> cascadeOutput(poolOut.size());
  for (size_t i = 0; i < poolOut.size(); ++i) {
    cascadeOutput[i] = poolOut[i] * scale;
  }

  // Run fused: Conv2D with fused filter
  std::vector<llvm::APFloat> filterAP;
  filterAP.reserve(filter.size());
  for (double v : filter) {
    filterAP.push_back(llvm::APFloat(v));
  }

  llvm::APFloat scaleAP(scale);
  auto fusedFilterAP = fuseFloatFilters(filterAP, conv, pool,
                                        llvm::APFloat::IEEEdouble(), scaleAP);

  std::vector<double> fusedFilter(fusedFilterAP.size());
  for (size_t i = 0; i < fusedFilterAP.size(); ++i) {
    fusedFilter[i] = fusedFilterAP[i].convertToDouble();
  }

  Conv2DSizeParams fusedConv = conv;
  fusedConv.kH = getEffectiveKernelHeight(conv, pool);
  fusedConv.kW = getEffectiveKernelWidth(conv, pool);
  fusedConv.strideH = conv.strideH * pool.strideH;
  fusedConv.strideW = conv.strideW * pool.strideW;
  fusedConv.dilationH = 1;
  fusedConv.dilationW = 1;

  auto fusedOutput = referenceConv2D(input, fusedFilter, fusedConv, inH, inW,
                                     poolOutH, poolOutW);

  EXPECT_THAT(fusedOutput,
              ::testing::Pointwise(::testing::DoubleNear(1e-5), cascadeOutput));
}

auto ConvDomain = fuzztest::StructOf<Conv2DSizeParams>(
    /*F=*/fuzztest::InRange(1L, 4L),
    /*C=*/fuzztest::InRange(1L, 4L),
    /*kH=*/fuzztest::InRange(1L, 4L),
    /*kW=*/fuzztest::InRange(1L, 4L),
    /*strideH=*/fuzztest::InRange(1L, 3L),
    /*strideW=*/fuzztest::InRange(1L, 3L),
    /*dilationH=*/fuzztest::InRange(1L, 3L),
    /*dilationW=*/fuzztest::InRange(1L, 3L));

auto PoolDomain = fuzztest::StructOf<Pool2DSizeParams>(
    /*pH=*/fuzztest::InRange(1L, 4L),
    /*pW=*/fuzztest::InRange(1L, 4L),
    /*strideH=*/fuzztest::InRange(1L, 3L),
    /*strideW=*/fuzztest::InRange(1L, 3L),
    /*dilationH=*/fuzztest::InRange(1L, 3L),
    /*dilationW=*/fuzztest::InRange(1L, 3L));

auto OutDomain = fuzztest::StructOf<OutputSizeParams>(
    /*poolOutH=*/fuzztest::InRange(1L, 4L),
    /*poolOutW=*/fuzztest::InRange(1L, 4L));

FUZZ_TEST(ConvPoolFusionTest, FuzzConvPoolEquivalence)
    .WithDomains(
        /*conv=*/ConvDomain,
        /*pool=*/PoolDomain,
        /*out=*/OutDomain,
        /*rawInput=*/fuzztest::VectorOf(fuzztest::InRange(-1.0, 1.0)),
        /*rawFilter=*/fuzztest::VectorOf(fuzztest::InRange(-1.0, 1.0)),
        /*scale=*/fuzztest::InRange(0.1, 2.0));

}  // namespace
}  // namespace heir
}  // namespace mlir
