#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;

struct BenchmarkResult {
  uint32_t ringDim;
  uint32_t numSlots;
  std::vector<uint32_t> levelBudget;
  uint32_t levelsAvailableAfterBootstrap;
  uint32_t dcrtBits;
  double setupTime;
  double evalTime;
  bool success;
};

BenchmarkResult run_benchmark(uint32_t ringDim,
                              uint64_t numSlots,
                              uint32_t levelsAvailableAfterBootstrap,
                              uint32_t dcrtBits,
                              std::vector<uint32_t> levelBudget) {
  BenchmarkResult result;
  result.numSlots = numSlots;
  result.levelBudget = levelBudget;
  result.levelsAvailableAfterBootstrap = levelsAvailableAfterBootstrap;
  result.dcrtBits = dcrtBits;
  result.success = false;
  result.setupTime = 0;
  result.evalTime = 0;
  result.ringDim = 0;

  try {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(dcrtBits + 1);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(numSlots);

    usint bootstrapDepth =
        FHECKKSRNS::GetBootstrapDepth(levelBudget, SPARSE_TERNARY);
    usint depth = levelsAvailableAfterBootstrap + bootstrapDepth;
    parameters.SetMultiplicativeDepth(depth);

    std::cout << "Generating CryptoContext for depth " << depth << "..."
              << std::endl;
    auto cryptoContext = GenCryptoContext(parameters);
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(FHE);

    result.ringDim = cryptoContext->GetRingDimension();

    std::cout << "Starting setup for ring dimension " << result.ringDim << " and "
              << numSlots << " slots..." << std::endl;
    auto startSetup = std::chrono::high_resolution_clock::now();

    std::cout << "  Running EvalBootstrapSetup..." << std::endl;
    cryptoContext->EvalBootstrapSetup(levelBudget, {0, 0}, numSlots);

    std::cout << "  Generating keys..." << std::endl;
    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);

    std::cout << "  Generating bootstrap keys..." << std::endl;
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    auto endSetup = std::chrono::high_resolution_clock::now();
    result.setupTime =
        std::chrono::duration<double>(endSetup - startSetup).count();
    std::cout << "Setup completed in " << result.setupTime << "s" << std::endl;

    std::vector<double> x = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1);
    ptxt->SetLength(numSlots);
    Ciphertext<DCRTPoly> ciph = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

    std::cout << "Running EvalBootstrap..." << std::endl;
    auto startEval = std::chrono::high_resolution_clock::now();
    auto ciphertextAfter = cryptoContext->EvalBootstrap(ciph);
    auto endEval = std::chrono::high_resolution_clock::now();

    result.evalTime =
        std::chrono::duration<double>(endEval - startEval).count();
    result.success = true;
    std::cout << "Bootstrap completed in " << result.evalTime << "s"
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Exception raised during benchmark" << std::endl;
  } catch (...) {
    std::cerr << "Unknown error during benchmark" << std::endl;
  }

  return result;
}

int main(int argc, char* argv[]) {
  std::string out_path = "openfhe_results.toml";
  if (argc > 1) out_path = argv[1];

  std::ofstream outfile(out_path);

  // Ring dimensions to test: 2^15 and 2^16 (matching Lattigo)
  std::vector<uint32_t> ringDims = {32768, 65536};

  for (uint32_t ringDim : ringDims) {
    uint32_t maxSlots = ringDim / 2;  // Dense slot usage

    // Test dense slot usage (comparable to Lattigo)
    std::cout << "\n--- Running benchmark for ringDim = " << ringDim
              << ", numSlots = " << maxSlots << " (dense) ---" << std::endl;
    BenchmarkResult res = run_benchmark(ringDim, maxSlots, 1, 40, {1, 1});
    if (res.success) {
      outfile << "[[results]]" << std::endl;
      outfile << "security_level = \"HEStd_128_classic\"" << std::endl;
      outfile << "ring_dim = " << res.ringDim << std::endl;
      outfile << "num_slots = " << res.numSlots << std::endl;
      outfile << "slot_usage = \"dense\"" << std::endl;
      outfile << "dcrt_bits = " << res.dcrtBits << std::endl;
      outfile << "level_budget = [" << res.levelBudget[0] << ", "
              << res.levelBudget[1] << "]" << std::endl;
      outfile << "levels_available_after_bootstrap = "
              << res.levelsAvailableAfterBootstrap << std::endl;
      outfile << "setup_latency_seconds = " << res.setupTime << std::endl;
      outfile << "eval_latency_seconds = " << res.evalTime << std::endl;
      outfile.flush();
    }

    // Test sparse slot usage (OpenFHE has different codepaths for sparse)
    for (uint32_t numSlots : {8, 1024}) {
      std::cout << "\n--- Running benchmark for ringDim = " << ringDim
                << ", numSlots = " << numSlots << " (sparse) ---" << std::endl;
      BenchmarkResult res = run_benchmark(ringDim, numSlots, 1, 40, {1, 1});
      if (res.success) {
        outfile << "[[results]]" << std::endl;
        outfile << "security_level = \"HEStd_128_classic\"" << std::endl;
        outfile << "ring_dim = " << res.ringDim << std::endl;
        outfile << "num_slots = " << res.numSlots << std::endl;
        outfile << "slot_usage = \"sparse\"" << std::endl;
        outfile << "dcrt_bits = " << res.dcrtBits << std::endl;
        outfile << "level_budget = [" << res.levelBudget[0] << ", "
                << res.levelBudget[1] << "]" << std::endl;
        outfile << "levels_available_after_bootstrap = "
                << res.levelsAvailableAfterBootstrap << std::endl;
        outfile << "setup_latency_seconds = " << res.setupTime << std::endl;
        outfile << "eval_latency_seconds = " << res.evalTime << std::endl;
        outfile.flush();
      }
    }
  }
  return 0;
}
