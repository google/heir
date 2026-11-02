load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "mont_runner",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "mont_runner_test.cc",
    mlir_src = "mont_runner.mlir",
    deps = [
        "@heir//lib/Utils/CheddarRef:mont_arith",
    ],
)
