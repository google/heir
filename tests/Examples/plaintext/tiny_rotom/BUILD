load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# One secret 1x16 vector through a public 16x8 layer plus bias, run in
# cleartext at several slot counts. Small enough to read, but at n = 64 and
# below the weights span several ciphertexts.
[
    llvm_runner_test(
        name = "tiny_rotom_%s_test" % n,
        heir_opt_flags = ["--mlir-to-rotom-plaintext=ciphertext-size=%s" % n],
        main_c_src = "tiny_matmul_test.cpp",
        mlir_src = "tiny_matmul.mlir",
    )
    for n in [
        "256",
        "64",
        "32",
    ]
]
