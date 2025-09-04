---
title: Getting Started
weight: 10
---

## Getting HEIR

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

### Via pip

We publish a python package [heir_py](https://pypi.org/project/heir-py/) that
includes the `heir-opt` and `heir-translate` binaries.

```
python -m venv venv
source venv/bin/activate

pip install heir_py

heir-opt --help
heir-translate --help
```

### Building From Source

#### Prerequisites

- [Git](https://git-scm.com/)
- A C++ compiler and linker ([clang](https://clang.llvm.org/) and
  [lld](https://lld.llvm.org/) or a recent version of `gcc`). If you want to run
  OpenFHE with parallelism (enabled by default), you'll also need OpenMP.
- Bazel via [bazelisk](https://github.com/bazelbuild/bazelisk). The precise
  Bazel version used is in `.bazelversion` in the repository root.

<details>
  <summary>Detailed Instructions</summary>
  The first two requirements are frequently pre-installed
  or can be installed via the system package manager.
  For example, on Ubuntu, these can be installed with

```bash
sudo apt-get update && sudo apt-get install clang lld libomp-dev
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
mkdir -p ~/bin
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
mv bazel ~/bin/bazel
```

Note that on linux systems, your OS user must **not** be `root` as `bazel` might
refuse to work if run as root.

On macOS, you can install `bazelisk` via Homebrew.

</details>

#### Clone and build the project

You can clone and build HEIR from the terminal as described below. Please see
[Development](https://heir.dev/docs/development) for information on IDE
configuration if you want to use an IDE to build HEIR.

```bash
git clone git@github.com:google/heir.git && cd heir
bazel build @heir//tools:heir-opt
```

Some HEIR passes require Yosys as a dependency (`--yosys-optimizer`), which
itself adds many transitive dependencies that may not build properly on all
systems. If you would like to skip Yosys and ABC compilation, use the following
build setting:

```bash
bazel build --//:enable_yosys=0 --build_tag_filters=-yosys @heir//tools:heir-opt
```

Adding the following to `.bazelrc` in the HEIR project root will make this the
default behavior

```
common --//:enable_yosys=0
common --build_tag_filters=-yosys
```

#### Optional: Run the tests

```bash
bazel test @heir//...
```

## Using HEIR

### Run the `dot-product` example

The `dot-product` program computes the dot product of two length-8 vectors of
16-bit integers (`i16` in MLIR parlance). This example will showcase the OpenFHE
backend by manually calling the relevant compiler passes and setting up a C++
harness to call into the HEIR-generated functions.

The input program is in `tests/Examples/common/dot_product_8.mlir`. Support for
standard input languages like `C` and `C++` are currently experimental at best,
but eventually we would use an MLIR-based tool to convert an input language to
MLIR like in that file. The program is below:

```mlir
func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16> {secret.secret}) -> i16 {
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

Now we run the `heir-opt` command to optimize and compile the program. If you
fetched a pre-built binary instead of building from source, then all commands
below should have `bazel run //tools:heir-opt --` replaced with `heir-opt`, and
similarly for `heir-translate`.

```bash
bazel run //tools:heir-opt -- \
--mlir-to-bgv='ciphertext-degree=8'\
--scheme-to-openfhe='entry-function=dot_product'  \
$PWD/tests/Examples/common/dot_product_8.mlir > output.mlir
```

This produces a file in the `openfhe` exit dialect (part of HEIR).

```mlir
!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
#polynomial_evaluation_encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L2_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**8>>
#ring_rns_L2_1_x8_ = #polynomial.ring<coefficientType = !rns_L2_, polynomialModulus = <1 + x**8>>
!rlwe_pt_L0_ = !lwe.rlwe_plaintext<encoding = #polynomial_evaluation_encoding, ring = #ring_rns_L0_1_x8_, underlying_type = i16>
!rlwe_pt_L1_ = !lwe.rlwe_plaintext<encoding = #polynomial_evaluation_encoding, ring = #ring_rns_L1_1_x8_, underlying_type = tensor<8xi16>>
!rlwe_pt_L2_ = !lwe.rlwe_plaintext<encoding = #polynomial_evaluation_encoding, ring = #ring_rns_L2_1_x8_, underlying_type = tensor<8xi16>>
#rlwe_params_L0_ = #lwe.rlwe_params<ring = #ring_rns_L0_1_x8_>
#rlwe_params_L1_ = #lwe.rlwe_params<ring = #ring_rns_L1_1_x8_>
#rlwe_params_L2_ = #lwe.rlwe_params<ring = #ring_rns_L2_1_x8_>
#rlwe_params_L2_D3_ = #lwe.rlwe_params<dimension = 3, ring = #ring_rns_L2_1_x8_>
!rlwe_ct_L0_ = !lwe.rlwe_ciphertext<encoding = #polynomial_evaluation_encoding, rlwe_params = #rlwe_params_L0_, underlying_type = i16>
!rlwe_ct_L1_ = !lwe.rlwe_ciphertext<encoding = #polynomial_evaluation_encoding, rlwe_params = #rlwe_params_L1_, underlying_type = tensor<8xi16>>
!rlwe_ct_L1_1 = !lwe.rlwe_ciphertext<encoding = #polynomial_evaluation_encoding, rlwe_params = #rlwe_params_L1_, underlying_type = i16>
!rlwe_ct_L2_ = !lwe.rlwe_ciphertext<encoding = #polynomial_evaluation_encoding, rlwe_params = #rlwe_params_L2_, underlying_type = tensor<8xi16>>
!rlwe_ct_L2_D3_ = !lwe.rlwe_ciphertext<encoding = #polynomial_evaluation_encoding, rlwe_params = #rlwe_params_L2_D3_, underlying_type = tensor<8xi16>>
module {
  func.func @dot_product(%arg0: !openfhe.crypto_context, %arg1: !rlwe_ct_L2_, %arg2: !rlwe_ct_L2_) -> !rlwe_ct_L0_ {
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
    %0 = openfhe.mul_no_relin %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !rlwe_ct_L2_, !rlwe_ct_L2_) -> !rlwe_ct_L2_D3_
    %1 = openfhe.relin %arg0, %0 : (!openfhe.crypto_context, !rlwe_ct_L2_D3_) -> !rlwe_ct_L2_
    %2 = openfhe.rot %arg0, %1 {index = 4 : index} : (!openfhe.crypto_context, !rlwe_ct_L2_) -> !rlwe_ct_L2_
    %3 = openfhe.add %arg0, %1, %2 : (!openfhe.crypto_context, !rlwe_ct_L2_, !rlwe_ct_L2_) -> !rlwe_ct_L2_
    %4 = openfhe.rot %arg0, %3 {index = 2 : index} : (!openfhe.crypto_context, !rlwe_ct_L2_) -> !rlwe_ct_L2_
    %5 = openfhe.add %arg0, %3, %4 : (!openfhe.crypto_context, !rlwe_ct_L2_, !rlwe_ct_L2_) -> !rlwe_ct_L2_
    %6 = openfhe.rot %arg0, %5 {index = 1 : index} : (!openfhe.crypto_context, !rlwe_ct_L2_) -> !rlwe_ct_L2_
    %7 = openfhe.add %arg0, %5, %6 : (!openfhe.crypto_context, !rlwe_ct_L2_, !rlwe_ct_L2_) -> !rlwe_ct_L2_
    %8 = openfhe.mod_reduce %arg0, %7 : (!openfhe.crypto_context, !rlwe_ct_L2_) -> !rlwe_ct_L1_
    %9 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<8xi64>) -> !rlwe_pt_L1_
    %10 = openfhe.mul_plain %arg0, %8, %9 : (!openfhe.crypto_context, !rlwe_ct_L1_, !rlwe_pt_L1_) -> !rlwe_ct_L1_
    %11 = openfhe.rot %arg0, %10 {index = 7 : index} : (!openfhe.crypto_context, !rlwe_ct_L1_) -> !rlwe_ct_L1_
    %12 = lwe.reinterpret_application_data %11 : !rlwe_ct_L1_ to !rlwe_ct_L1_1
    %13 = openfhe.mod_reduce %arg0, %12 : (!openfhe.crypto_context, !rlwe_ct_L1_1) -> !rlwe_ct_L0_
    return %13 : !rlwe_ct_L0_
  }
  func.func @dot_product__encrypt__arg0(%arg0: !openfhe.crypto_context, %arg1: tensor<8xi16>, %arg2: !openfhe.public_key) -> !rlwe_ct_L2_ {
    ...
  }
  func.func @dot_product__encrypt__arg1(%arg0: !openfhe.crypto_context, %arg1: tensor<8xi16>, %arg2: !openfhe.public_key) -> !rlwe_ct_L2_ {
    ...
  }
  func.func @dot_product__decrypt__result0(%arg0: !openfhe.crypto_context, %arg1: !rlwe_ct_L0_, %arg2: !openfhe.private_key) -> i16 {
    ...
  }
  func.func @dot_product__generate_crypto_context() -> !openfhe.crypto_context {
    ...
  }
  func.func @dot_product__configure_crypto_context(%arg0: !openfhe.crypto_context, %arg1: !openfhe.private_key) -> !openfhe.crypto_context {
    ...
  }
}
```

Next, we use the `heir-translate` tool to run code generation for the OpenFHE
`pke` API.

```bash
bazel run //tools:heir-translate -- --emit-openfhe-pke-header --openfhe-include-type=source-relative $PWD/output.mlir > heir_output.h
bazel run //tools:heir-translate -- --emit-openfhe-pke --openfhe-include-type=source-relative $PWD/output.mlir > heir_output.cpp
```

The `openfhe-include-type` indicates which include path for OpenFHE is used. It
has three possible values: `install-relative`, `source-relative` and `embedded`.
In this example we use `source-relative` as we are compiling against an
(unoptimized) OpenFHE managed by bazel in HEIR source. To compile against an
installed (and possibly optimized) OpenFHE, you could use `install-relative` and
compile it on your own. Or you could just put the generated file in OpenFHE
source directory `src/pke/examples` and let OpenFHE find and compile it for you
with the `embedded` option.

The results:

```cpp
// heir_output.h
#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextBGVRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT dot_product(CryptoContextT v0, CiphertextT v1, CiphertextT v2);
CiphertextT dot_product__encrypt__arg0(CryptoContextT v18, std::vector<int16_t> v19, PublicKeyT v20);
CiphertextT dot_product__encrypt__arg1(CryptoContextT v24, std::vector<int16_t> v25, PublicKeyT v26);
int16_t dot_product__decrypt__result0(CryptoContextT v30, CiphertextT v31, PrivateKeyT v32);
CryptoContextT dot_product__generate_crypto_context();
CryptoContextT dot_product__configure_crypto_context(CryptoContextT v37, PrivateKeyT v38);


// heir_output.cpp
#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT dot_product(CryptoContextT v0, CiphertextT v1, CiphertextT v2) {
  std::vector<int64_t> v3 = {0, 0, 0, 0, 0, 0, 0, 1};
  const auto& v4 = v0->EvalMultNoRelin(v1, v2);
  const auto& v5 = v0->Relinearize(v4);
  const auto& v6 = v0->EvalRotate(v5, 4);
  const auto& v7 = v0->EvalAdd(v5, v6);
  const auto& v8 = v0->EvalRotate(v7, 2);
  const auto& v9 = v0->EvalAdd(v7, v8);
  const auto& v10 = v0->EvalRotate(v9, 1);
  const auto& v11 = v0->EvalAdd(v9, v10);
  const auto& v12 = v0->ModReduce(v11);
  auto v3_filled_n = v0->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v3_filled = v3;
  v3_filled.clear();
  v3_filled.reserve(v3_filled_n);
  for (auto i = 0; i < v3_filled_n; ++i) {
    v3_filled.push_back(v3[i % v3.size()]);
  }
  const auto& v13 = v0->MakePackedPlaintext(v3_filled);
  const auto& v14 = v0->EvalMult(v12, v13);
  const auto& v15 = v0->EvalRotate(v14, 7);
  const auto& v16 = v15;
  const auto& v17 = v0->ModReduce(v16);
  return v17;
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
CryptoContextT dot_product__generate_crypto_context() {
  ...
}
CryptoContextT dot_product__configure_crypto_context(CryptoContextT v37, PrivateKeyT v38) {
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

#include "src/pke/include/openfhe.h"  // from @openfhe
#include "heir_output.h"

int main(int argc, char *argv[]) {
  CryptoContext<DCRTPoly> cryptoContext = dot_product__generate_crypto_context();

  KeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();

  cryptoContext = dot_product__configure_crypto_context(cryptoContext, keyPair.secretKey);

  std::vector<int16_t> arg0 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int16_t> arg1 = {2, 3, 4, 5, 6, 7, 8, 9};
  int64_t expected = 240;

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

If you fetched a pre-built binary instead of building from source, then you will
have to use your build system of choice to compile the generated files. If you
use `heir_py`'s `heir.compile` decorator with `debug=True`, then the compilation
commands will be printed to stdout so you can see how to compile the generated
code manually.

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
  --secret-to-cggi -cse \
  $PWD/tests/Dialect/Secret/Conversions/secret_to_cggi/add_one.mlir
```

To convert an existing lit test to a `bazel run` command for manual tweaking and
introspection (e.g., adding `--debug` or `--mlir-print-ir-after-all` to see how
he IR changes with each pass), use `python scripts/lit_to_bazel.py`.

```bash
# after pip install -r requirements.txt
python scripts/lit_to_bazel.py tests/simd/box_blur_64x64.mlir
```

Which outputs

```bash
bazel run --noallow_analysis_cache_discard //tools:heir-opt -- \
--secretize --wrap-generic --canonicalize --cse --full-loop-unroll \
--insert-rotate --cse --canonicalize --collapse-insertion-chains \
--canonicalize --cse /path/to/heir/tests/simd/box_blur_64x64.mlir
```

### Optional: Graphviz visualization of the IR

Getting a visualization of the IR during optimization/transformation might help
you understand what is going on more easily.

Still taking the `dot_product_8.mlir` as an example:

```bash
bazel run --ui_event_filters=-info,-debug,-warning,-stderr,-stdout --noshow_progress --logging=0 //tools:heir-opt -- --wrap-generic --heir-simd-vectorizer $PWD/tests/Examples/common/dot_product_8.mlir --view-op-graph 2> dot_product_8.dot
dot -Tpdf dot_product_8.dot > dot_product_8.pdf
# open pdf in your favorite pdf viewer
```

The diagram is also shown below. It demonstrates that the HEIR SIMD vectorizer
would vectorize the dot-product program (`tensor<8xi16>`) then use
rotate-and-reduce technique to compute the sum.

{{% figure src="/images/dot_product_8.svg" link="/images/dot_product_8.svg" %}}
