load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "loop",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=8 insert-debug-handler-calls=true",
    ],
    log_file_name = "loop_support_debug.log",
    log_file_visibility = ["@heir//tests/Examples:__subpackages__"],
    main_c_src = "loop_test.cpp",
    mlir_src = "@heir//tests/Examples/common:loop.mlir",
)
