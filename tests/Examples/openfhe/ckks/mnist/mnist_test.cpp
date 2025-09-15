#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/core/include/utils/inttypes.h"           // from @openfhe
#include "src/pke/include/key/keypair.h"               // from @openfhe
#include "src/pke/include/key/privatekey-fwd.h"        // from @openfhe

// Block clang-format from reordering
// clang-format off
#include "tools/cpp/runfiles/runfiles.h"  // from @bazel_tools
#include "gtest/gtest.h"  // from @googletest
#include "torch/csrc/api/include/torch/data/dataloader.h"  // from @torch
#include "torch/csrc/api/include/torch/data/datasets/base.h"  // from @torch
#include "torch/csrc/api/include/torch/data/datasets/map.h"  // from @torch
#include "torch/csrc/api/include/torch/data/samplers/sequential.h"  // from @torch
#include "torch/csrc/api/include/torch/data/transforms/stack.h"  // from @torch
#include "torch/csrc/api/include/torch/data/transforms/tensor.h"  // from @torch
#include "torch/csrc/jit/api/module.h"  // from @torch
#include "torch/csrc/jit/serialization/import.h"  // from @torch
#include "torch/data/datasets/mnist.h"  // from @torch
// clang-format on

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/mnist/mnist_lib.h"

using bazel::tools::cpp::runfiles::Runfiles;

constexpr std::string_view kModelPath =
    "/heir/tests/Examples/openfhe/ckks/mnist/data/"
    "traced_model.pt";
constexpr std::string_view kDataPath =
    "/heir/tests/Examples/openfhe/ckks/mnist/data";

namespace mlir {
namespace heir {
namespace openfhe {

namespace {

template <int N>
int argmax(float* A) {
  int max_idx = 0;
  for (int i = 1; i < N; i++) {
    if (A[i] > A[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

std::vector<std::vector<float>> loadWeights(const std::string& modelPath) {
  std::vector<std::vector<float>> weights;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load(modelPath);
    module.eval();  // Don't forget to set evaluation mode

    std::cout << "Successfully loaded " << modelPath << std::endl;

    // Access and print parameter names and shapes
    std::cout << "Model parameters:" << std::endl;
    for (auto pair : module.named_parameters()) {
      const std::string& name = pair.name;
      const auto& tensor = pair.value;
      std::cout << "  " << name << ": " << tensor.sizes() << std::endl;

      auto tensorCont = tensor.contiguous();
      int64_t num_elements = tensorCont.numel();
      const float* tensor_data = tensorCont.data_ptr<float>();

      weights.push_back({tensor_data, tensor_data + num_elements});
    }
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model" << std::endl;
    return {};
  }
  return weights;
}

}  // namespace

TEST(MNISTTest, RunTest) {
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest());

  auto weights = loadWeights(runfiles->Rlocation(kModelPath));
  ASSERT_FALSE(weights.empty());

  auto test_dataset =
      torch::data::datasets::MNIST(runfiles->Rlocation(kDataPath),
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), 1);

  auto cryptoContext = mnist__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mnist__configure_crypto_context(cryptoContext, secretKey);

  std::cout << *cryptoContext->GetCryptoParameters() << std::endl;

  int total = 4;
  int correct = 0;
  for (auto& batch : *test_loader) {
    if (total == 0) break;
    auto input_tensor = batch.data.contiguous();
    float* tensor_data_ptr = input_tensor.data_ptr<float>();

    std::vector<float> input_vector(tensor_data_ptr,
                                    tensor_data_ptr + input_tensor.numel());
    auto input_encrypted =
        mnist__encrypt__arg4(cryptoContext, input_vector, publicKey);

    std::clock_t cStart = std::clock();
    auto output_encrypted = mnist(cryptoContext, weights[0], weights[1],
                                  weights[2], weights[3], input_encrypted);
    std::clock_t cEnd = std::clock();

    double timeElapsedMs = 1000.0 * (cEnd - cStart) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << timeElapsedMs << " ms\n";

    std::vector<float> output =
        mnist__decrypt__result0(cryptoContext, output_encrypted, secretKey);
    auto label_tensor = batch.target;
    int64_t label = label_tensor.item<int64_t>();
    auto max_id = argmax<10>(output.data());

    if (max_id == label) {
      correct++;
    }
    std::cout << "max_id: " << max_id << ", label: " << label << std::endl;
    total--;
  }

  EXPECT_GE(correct, 0.9 * total);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
