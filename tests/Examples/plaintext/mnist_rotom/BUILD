load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# Real MNIST through the Rotom pipeline, executed in cleartext: the
# torch-exported model of tests/Examples/common/mnist -- 1x784 secret image,
# 512x784 and 10x512 public layers with biases, degree-4 Chebyshev ReLU --
# seeded, searched, lowered, and run against a cleartext reference computing
# the same polynomial network. Weights are synthetic small-magnitude values
# (the check proves the compiler, not the model).
llvm_runner_test(
    name = "mnist_rotom_test",
    heir_opt_flags = [
        "--mlir-to-rotom-plaintext=ciphertext-size=32768",
    ],
    main_c_src = "mnist_rotom_test.cpp",
    mlir_src = "@heir//tests/Examples/common/mnist:mnist.mlir",
)
