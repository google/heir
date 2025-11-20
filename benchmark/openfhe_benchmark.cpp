// HEIR note: this is ported from OpenFHE in order to ensure that our bazel
// build produces similar timing numbers as the upstream OpenFHE cmake build.
// https://github.com/openfheorg/openfhe-development/blob/aa391988d354d4360f390f223a90e0d1b98839d7/benchmark/src/lib-benchmark.cpp

#include <complex>
#include <cstdint>
#include <vector>

#define _USE_MATH_DEFINES

#include "benchmark/benchmark.h"                       // from @google_benchmark
#include "src/core/include/lattice/stdlatticeparms.h"  // from @openfhe
#include "src/core/include/math/hal/basicint.h"        // from @openfhe
#include "src/pke/include/cryptocontext.h"             // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"         // from @openfhe
#include "src/pke/include/scheme/ckksrns/gen-cryptocontext-ckksrns.h"  // from @openfhe

using namespace lbcrypto;

[[maybe_unused]] static void DepthArgs(benchmark::internal::Benchmark* b) {
  for (uint32_t d : {1, 2, 4, 6, 8, 10, 12}) b->ArgName("depth")->Arg(d);
}

[[maybe_unused]] static CryptoContext<DCRTPoly> GenerateCKKSContext(
    uint32_t mdepth = 1) {
  CCParams<CryptoContextCKKSRNS> parameters;
  // parameters.SetSecurityLevel(HEStd_NotSet);
  parameters.SetScalingTechnique(FIXEDMANUAL);
  parameters.SetRingDim(8192);
  parameters.SetScalingModSize(48);
  parameters.SetMultiplicativeDepth(mdepth);
  auto cc = GenCryptoContext(parameters);
  // std::cout << "Ring dim: " << cc->GetRingDimension() << "\n";
  // Default ring dimension is 16384 when nothing is set
  // Default ring dimension is 8192 when FIXEDMANUAL is set
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}

void CKKSrns_Add(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  while (state.KeepRunning()) {
    auto ciphertextAdd = cc->EvalAdd(ciphertext1, ciphertext2);
  }
}

void CKKSrns_AddInPlace(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  while (state.KeepRunning()) {
    cc->EvalAddInPlace(ciphertext1, ciphertext2);
  }
}

void CKKSrns_KeyGen(benchmark::State& state) {
  CryptoContext<DCRTPoly> cryptoContext = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair;

  while (state.KeepRunning()) {
    keyPair = cryptoContext->KeyGen();
  }
}

void CKKSrns_MultKeyGen(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair;
  keyPair = cc->KeyGen();

  while (state.KeepRunning()) {
    cc->EvalMultKeyGen(keyPair.secretKey);
  }
}

void CKKSrns_EvalAtIndexKeyGen(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair;
  keyPair = cc->KeyGen();

  std::vector<int32_t> indexList(1);
  for (usint i = 0; i < 1; i++) {
    indexList[i] = 1;
  }

  while (state.KeepRunning()) {
    cc->EvalAtIndexKeyGen(keyPair.secretKey, indexList);
  }
}

void CKKSrns_Encryption(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts[i] = 1.001 * i;
  }

  auto plaintext = cc->MakeCKKSPackedPlaintext(vectorOfInts);

  while (state.KeepRunning()) {
    auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);
  }
}

void CKKSrns_Decryption(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  ciphertext1 = cc->LevelReduce(ciphertext1, nullptr, 1);

  Plaintext plaintextDec1;

  while (state.KeepRunning()) {
    cc->Decrypt(keyPair.secretKey, ciphertext1, &plaintextDec1);
  }
}

void CKKSrns_MultNoRelin(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext(state.range(0));

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  while (state.KeepRunning()) {
    auto ciphertextMul = cc->EvalMultNoRelin(ciphertext1, ciphertext2);
  }

  state.SetComplexityN(state.range(0));
}

void CKKSrns_MultRelin(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext(state.range(0));

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  while (state.KeepRunning()) {
    auto ciphertextMul = cc->EvalMult(ciphertext1, ciphertext2);
  }

  state.SetComplexityN(state.range(0));
}

void CKKSrns_Relin(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  auto ciphertextMul = cc->EvalMultNoRelin(ciphertext1, ciphertext2);

  while (state.KeepRunning()) {
    auto ciphertext3 = cc->Relinearize(ciphertextMul);
  }
}

void CKKSrns_RelinInPlace(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  auto ciphertextMul = cc->EvalMultNoRelin(ciphertext1, ciphertext2);
  auto ciphertextMulClone = ciphertextMul->Clone();

  while (state.KeepRunning()) {
    cc->RelinearizeInPlace(ciphertextMul);
    state.PauseTiming();
    ciphertextMul = ciphertextMulClone->Clone();
    state.ResumeTiming();
  }
}

void CKKSrns_Rescale(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  auto ciphertextMul = cc->EvalMult(ciphertext1, ciphertext2);

  while (state.KeepRunning()) {
    auto ciphertext3 = cc->ModReduce(ciphertextMul);
  }
}

void CKKSrns_RescaleInPlace(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts[i] = 1.001 * i;
  }

  auto plaintext = cc->MakeCKKSPackedPlaintext(vectorOfInts);
  auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);
  auto ciphertextMul = cc->EvalMult(ciphertext, ciphertext);
  auto ciphertextMulClone = ciphertextMul->Clone();

  while (state.KeepRunning()) {
    cc->ModReduceInPlace(ciphertextMul);
    state.PauseTiming();
    ciphertextMul = ciphertextMulClone->Clone();
    state.ResumeTiming();
  }
}

void CKKSrns_EvalAtIndex(benchmark::State& state) {
  CryptoContext<DCRTPoly> cc = GenerateCKKSContext();

  KeyPair<DCRTPoly> keyPair = cc->KeyGen();
  cc->EvalMultKeyGen(keyPair.secretKey);

  std::vector<int32_t> indexList(1);
  for (usint i = 0; i < 1; i++) {
    indexList[i] = 1;
  }

  cc->EvalAtIndexKeyGen(keyPair.secretKey, indexList);

  usint slots = cc->GetEncodingParams()->GetBatchSize();
  std::vector<std::complex<double>> vectorOfInts1(slots);
  for (usint i = 0; i < slots; i++) {
    vectorOfInts1[i] = 1.001 * i;
  }
  std::vector<std::complex<double>> vectorOfInts2(vectorOfInts1);

  auto plaintext1 = cc->MakeCKKSPackedPlaintext(vectorOfInts1);
  auto plaintext2 = cc->MakeCKKSPackedPlaintext(vectorOfInts2);

  auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
  auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

  auto ciphertextMul = cc->EvalMult(ciphertext1, ciphertext2);

  while (state.KeepRunning()) {
    auto ciphertext3 = cc->EvalAtIndex(ciphertextMul, 1);
  }
}

BENCHMARK(CKKSrns_Add)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_AddInPlace)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_Decryption)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_Encryption)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_EvalAtIndex)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_EvalAtIndexKeyGen)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_KeyGen)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_MultKeyGen)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_MultNoRelin)->Unit(benchmark::kMicrosecond)->Apply(DepthArgs);
BENCHMARK(CKKSrns_MultRelin)->Unit(benchmark::kMicrosecond)->Apply(DepthArgs);
BENCHMARK(CKKSrns_Relin)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_RelinInPlace)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_Rescale)->Unit(benchmark::kMicrosecond);
BENCHMARK(CKKSrns_RescaleInPlace)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
