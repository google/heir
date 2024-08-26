---
title: Getting Started
weight: 10
---

## Getting HEIR

### Building From Source

#### Prerequisites

- [Git](https://git-scm.com/)
- A C++ compiler and linker ([clang](https://clang.llvm.org/) and
  [lld](https://lld.llvm.org/) are recommended).
- Bazel via [bazelisk](https://github.com/bazelbuild/bazelisk), or version
  `>=5.5`
- See [Development](https://heir.dev/docs/development) for additional
  prerequisites for active development

<details>
  <summary>Detailed Instructions</summary>
  The first two requirements are frequently pre-installed
  or can be installed via the system package manager.
  For example, on Ubuntu, these can be installed with

```bash
sudo apt-get update && sudo apt-get install clang lld
```

You can download the latest Bazelisk release, e.g., for linux-amd64 (see the
[Bazelisk Release Page](https://github.com/bazelbuild/bazelisk/releases/latest)
for a list of available binaries):

```bash
wget -c https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
mv bazelisk-linux-amd64 bazel
chmod +x bazel
```

You will then likely want to move `bazel` to a location on your PATH, or add its
location to your PATH, e.g.:

```bash
mkdir ~/bin
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
mv bazel ~/bin/bazel
```

Note that on linux systems, your OS user must **not** be `root` as bazel might
refuse to work if run as root.

</details>

#### Clone and build the project

You can clone and build HEIR from the terminal as described below. Please see
[Development](https://heir.dev/docs/development) for information on IDE
configuration if you want to use an IDE to build HEIR.

```bash
git clone git@github.com:google/heir.git && cd heir
bazel build @heir//tools:heir-opt
```

Some passes in this repository require Yosys as a dependency
(`--yosys-optimizer`). If you would like to skip Yosys and ABC compilation to
speed up builds, use the following build setting:

```bash
bazel build --//:enable_yosys=0 @heir//tools:heir-opt
```

#### Optional: Run the tests

```bash
bazel test @heir//...
```

Like above, run the following to skip tests that depend on Yosys:

```bash
bazel test --//:enable_yosys=0 --test_tag_filters=-yosys @heir//...
```

### Using a pre-built nightly binary

HEIR releases a [nightly](https://github.com/google/heir/releases/tag/nightly)
binary for Linux x86-64. This is intended for testing compiler passes and not
for production use.

```bash
wget https://github.com/google/heir/releases/download/nightly/heir-opt
chmod +x heir-opt
./heir-opt --help
```

Then you can run the examples below, replacing `bazel run //tools:heir-opt --`
with `./heir-opt`. HEIR also publishes `heir-translate` and `heir-lsp` in the
same way.

## Using HEIR

### Run the `dot-product` example

The `dot-product` program computes the dot product of two length-8 vectors of
16-bit integers (`i16` in MLIR parlance). This example will showcase the OpenFHE
backend by manually calling the relevant compiler passes and setting up a C++
harness to call into the HEIR-generated functions.

Note: other backends are similar, but the different backends are in varying
stages of development.

The input program is in `tests/openfhe/end_to_end/dot_product_8.mlir`. Support
for standard input languages like `C` and `C++` are currently experimental at
best, but eventually we would use an MLIR-based tool to convert an input
language to MLIR like in that file. The program is below:

```mlir
func.func @dot_product(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 8 iter_args(%iter = %c0_si16) -> (i16) {
    %1 = tensor.extract %arg0[%arg2] : tensor<8xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<8xi16>
    %3 = arith.muli %1, %2 : i16
    %4 = arith.addi %iter, %3 : i16
    affine.yield %4 : i16
  }
  return %0 : i16
}
```

For an introduction to MLIR syntax, see the
[official docs](https://mlir.llvm.org/docs/LangRef/) or
[this blog post](https://www.jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/).

Now we run the `heir-opt` command to optimize and compile the program.

```bash
bazel run //tools:heir-opt -- \
--mlir-to-openfhe-bgv='entry-function=dot_product ciphertext-degree=8' \
$PWD/tests/openfhe/end_to_end/dot_product_8.mlir > output.mlir
```

This produces a file in the `openfhe` exit dialect (part of HEIR). The raw
output is rather verbose, and an abbreviated version is shown below.

```mlir
!tensor_ct = !lwe.rlwe_ciphertext<..., underlying_type = tensor<8xi16>>
!scalar_ct = !lwe.rlwe_ciphertext<..., underlying_type = i16>
!mul_ct = !lwe.rlwe_ciphertext<..., underlying_type = tensor<8xi16>>
!tensor_plaintext = lwe.rlwe_plaintext<..., underlying_type = tensor<8xi16>>
module {
  func.func @dot_product(%arg0: !openfhe.crypto_context, %arg1: !tensor_ct, %arg2: !tensor_ct) -> !scalar_ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %0 = openfhe.mul_no_relin %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !tensor_ct, !tensor_ct) -> !mul_ct
    %1 = openfhe.relin %arg0, %0 : (!openfhe.crypto_context, !mul_ct) -> !tensor_ct
    %2 = arith.index_cast %c4 : index to i64
    %3 = openfhe.rot %arg0, %1, %2 : (!openfhe.crypto_context, !tensor_ct, i64) -> !tensor_ct
    %4 = openfhe.add %arg0, %1, %3 : (!openfhe.crypto_context, !tensor_ct, !tensor_ct) -> !tensor_ct
    %5 = arith.index_cast %c2 : index to i64
    %6 = openfhe.rot %arg0, %4, %5 : (!openfhe.crypto_context, !tensor_ct, i64) -> !tensor_ct
    %7 = openfhe.add %arg0, %4, %6 : (!openfhe.crypto_context, !tensor_ct, !tensor_ct) -> !tensor_ct
    %8 = arith.index_cast %c1 : index to i64
    %9 = openfhe.rot %arg0, %7, %8 : (!openfhe.crypto_context, !tensor_ct, i64) -> !tensor_ct
    %10 = openfhe.add %arg0, %7, %9 : (!openfhe.crypto_context, !tensor_ct, !tensor_ct) -> !tensor_ct
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi16>
    %11 = lwe.rlwe_encode %cst {encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, ring = #_polynomial.ring<cmod=463187969, ideal=#_polynomial.polynomial<1 + x**8>>} : tensor<8xi16> -> !tensor_plaintext
    %12 = openfhe.mul_plain %arg0, %10, %11 : (!openfhe.crypto_context, !tensor_ct, !tensor_plaintext) -> !tensor_ct
    %13 = arith.index_cast %c7 : index to i64
    %14 = openfhe.rot %arg0, %12, %13 : (!openfhe.crypto_context, !tensor_ct, i64) -> !tensor_ct
    %15 = lwe.reinterpret_underlying_type %14 : !tensor_ct to !scalar_ct
    return %15 : !scalar_ct
  }
  func.func @dot_product__encrypt__arg0(%arg0: !openfhe.crypto_context, %arg1: tensor<8xi16>, %arg2: !openfhe.public_key) -> !tensor_ct
    ...
  }
  func.func @dot_product__encrypt__arg1(%arg0: !openfhe.crypto_context, %arg1: tensor<8xi16>, %arg2: !openfhe.public_key) -> !tensor_ct
    ...
  }
  func.func @dot_product__decrypt__result0(%arg0: !openfhe.crypto_context, %arg1: !scalar_ct, %arg2: !openfhe.private_key) -> i16 {
    ...
  }
}
```

Next, we use the `heir-translate` tool to run code generation for the OpenFHE
`pke` API.

```bash
bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/output.mlir > heir_output.h
bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/output.mlir > heir_output.cpp
```

The results:

```cpp
// heir_output.h
#include "src/pke/include/openfhe.h" // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT dot_product(CryptoContextT v0, CiphertextT v1, CiphertextT v2);
CiphertextT dot_product__encrypt__arg0(CryptoContextT v24, std::vector<int16_t> v25, PublicKeyT v26);
CiphertextT dot_product__encrypt__arg1(CryptoContextT v29, std::vector<int16_t> v30, PublicKeyT v31);
int16_t dot_product__decrypt__result0(CryptoContextT v34, CiphertextT v35, PrivateKeyT v36);

// heir_output.cpp
#include "src/pke/include/openfhe.h" // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT dot_product(CryptoContextT v0, CiphertextT v1, CiphertextT v2) {
  size_t v3 = 1;
  size_t v4 = 2;
  size_t v5 = 4;
  size_t v6 = 7;
  const auto& v7 = v0->EvalMultNoRelin(v1, v2);
  const auto& v8 = v0->Relinearize(v7);
  int64_t v9 = static_cast<int64_t>(v5);
  const auto& v10 = v0->EvalRotate(v8, v9);
  const auto& v11 = v0->EvalAdd(v8, v10);
  int64_t v12 = static_cast<int64_t>(v4);
  const auto& v13 = v0->EvalRotate(v11, v12);
  const auto& v14 = v0->EvalAdd(v11, v13);
  int64_t v15 = static_cast<int64_t>(v3);
  const auto& v16 = v0->EvalRotate(v14, v15);
  const auto& v17 = v0->EvalAdd(v14, v16);
  std::vector<int16_t> v18 = {0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<int64_t> v18_cast(std::begin(v18), std::end(v18));
  const auto& v19 = v0->MakePackedPlaintext(v18_cast);
  const auto& v20 = v0->EvalMult(v17, v19);
  int64_t v21 = static_cast<int64_t>(v6);
  const auto& v22 = v0->EvalRotate(v20, v21);
  const auto& v23 = v22;
  return v23;
}
CiphertextT dot_product__encrypt__arg0(CryptoContextT v24, std::vector<int16_t> v25, PublicKeyT v26) {
  ...
}
CiphertextT dot_product__encrypt__arg1(CryptoContextT v29, std::vector<int16_t> v30, PublicKeyT v31) {
  ...
}
int16_t dot_product__decrypt__result0(CryptoContextT v34, CiphertextT v35, PrivateKeyT v36) {
  ...
}
```

At this point we can compile the program as we would a normal OpenFHE program.
Note that the above two files just contain the compiled function and
encryption/decryption helpers, and does not include any code that provides
specific inputs or calls these functions.

Next we'll create a harness that provides sample inputs, encrypts them, runs the
compiled function, and decrypts the result. Once you have the generated header
and cpp files, you can do this with any build system. We will use bazel for
consistency.

Create a file called `BUILD` in the same directory as the header and cpp files
above, with the following contents:

```BUILD
# A library build target that encapsulates the HEIR-generated code.
cc_library(
    name = "dot_product_codegen",
    srcs = ["heir_output.cpp"],
    hdrs = ["heir_output.h"],
    deps = ["@openfhe//:pke"],
)

# An executable build target that contains your main function and links
# against the above.
cc_binary(
    name = "dot_product_main",
    srcs = ["dot_product_main.cpp"],
    deps = [
        ":dot_product_codegen",
        "@openfhe//:pke",
        "@openfhe//:core",
    ],
)
```

Where `dot_product_main.cpp` is a new file containing

```cpp
#include <cstdint>
#include <vector>

#include "src/pke/include/openfhe.h" // from @openfhe
#include "heir_output.h"

int main(int argc, char *argv[]) {
  CCParams<CryptoContextBGVRNS> parameters;
  // TODO(#661): replace this setup with a HEIR-generated helper function
  parameters.SetMultiplicativeDepth(2);
  parameters.SetPlaintextModulus(65537);
  CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);

  KeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);
  cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, 4, 7});

  int32_t n = cryptoContext->GetCryptoParameters()
                  ->GetElementParams()
                  ->GetCyclotomicOrder() /
              2;
  int16_t arg0Vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t arg1Vals[8] = {2, 3, 4, 5, 6, 7, 8, 9};
  int64_t expected = 240;

  std::vector<int16_t> arg0;
  std::vector<int16_t> arg1;
  arg0.reserve(n);
  arg1.reserve(n);

  // TODO(#645): support cyclic repetition in add-client-interface
  for (int i = 0; i < n; ++i) {
    arg0.push_back(arg0Vals[i % 8]);
    arg1.push_back(arg1Vals[i % 8]);
  }

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, keyPair.publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg1(cryptoContext, arg1, keyPair.publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, arg0Encrypted, arg1Encrypted);
  auto actual = dot_product__decrypt__result0(cryptoContext, outputEncrypted,
                                              keyPair.secretKey);

  std::cout << "Expected: " << expected << "\n";
  std::cout << "Actual: " << actual << "\n";

  return 0;
}
```

Then run and show the results:

```bash
$ bazel run dot_product_main
Expected: 240
Actual: 240
```

### Optional: Run a custom `heir-opt` pipeline

HEIR comes with two central binaries, `heir-opt` for running optimization passes
and dialect conversions, and `heir-translate` for backend code generation. To
see the list of available passes in each one, run the binary with `--help`:

```bash
bazel run //tools:heir-opt -- --help
bazel run //tools:heir-translate -- --help
```

Once you've chosen a pass or `--pass-pipeline` to run, execute it on the desired
file. For example, you can run a test file through `heir-opt` to see its output.
Note that when the binary is run via `bazel`, you must pass absolute paths to
input files. You can also access the underlying binary at
`bazel-bin/tools/heir-opt`, provided it has already been built.

```bash
bazel run //tools:heir-opt -- \
  --comb-to-cggi -cse \
  $PWD/tests/comb_to_cggi/add_one.mlir
```

To convert an existing lit test to a `bazel run` command for manual tweaking and
introspection (e.g., adding `--debug` or `--mlir-print-ir-after-all` to see how
he IR changes with each pass), use `python scripts/lit_to_bazel.py`.

```bash
# after pip installing requirements-dev.txt
python scripts/lit_to_bazel.py tests/simd/box_blur_64x64.mlir
```

Which outputs

```bash
bazel run --noallow_analysis_cache_discard //tools:heir-opt -- \
--secretize=entry-function=box_blur --wrap-generic --canonicalize --cse --full-loop-unroll \
--insert-rotate --cse --canonicalize --collapse-insertion-chains \
--canonicalize --cse /path/to/heir/tests/simd/box_blur_64x64.mlir
```
