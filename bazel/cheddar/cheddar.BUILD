load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "cheddar_cuda",
    srcs = [
        "src/UserInterface.cu",
        "src/core/ElementWise.cu",
        "src/core/ModSwitch.cu",
        "src/core/NTT.cu",
        "src/core/Parameter.cu",
        "src/extension/Hoist.cu",
    ],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
    ]),
    defines = [
        "ENABLE_EXTENSION",
        "SPDLOG_FMT_EXTERNAL",
        "_ALLOW_UNSUPPORTED_LIBCPP",
        "_LIBCPP_NO_ABI_TAG",
        "_VSTD=std",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@cuda//:cccl_headers",
        "@cuda//:cudart_headers",
        "@cuda//:thrust",
        "@libtommath//:tommath",
        "@rmm",
        "@rules_cuda//cuda:runtime",
    ],
)

cc_library(
    name = "cheddar",
    srcs = [
        "src/core/BigInt_tommath.cpp",
        "src/core/Container.cpp",
        "src/core/Context.cpp",
        "src/core/DeviceVector.cpp",
        "src/core/Encode.cpp",
        "src/core/EvkMap.cpp",
        "src/core/EvkRequest.cpp",
        "src/core/MemoryPool.cpp",
        "src/core/MultiLevelCiphertext.cpp",
        "src/core/NPInfo.cpp",
        "src/extension/BootContext.cpp",
        "src/extension/BootParameter.cpp",
        "src/extension/EvalMod.cpp",
        "src/extension/EvalPoly.cpp",
        "src/extension/EvalSpecialFFT.cpp",
        "src/extension/LinearTransform.cpp",
        "src/extension/StripedMatrix.cpp",
    ],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
    ]),
    defines = [
        "ENABLE_EXTENSION",
        "SPDLOG_FMT_EXTERNAL",
        "_ALLOW_UNSUPPORTED_LIBCPP",
        "_LIBCPP_NO_ABI_TAG",
        "_VSTD=std",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":cheddar_cuda",
        "@cuda//:cccl_headers",
        "@cuda//:cudart_headers",
        "@cuda//:thrust",
        "@libtommath//:tommath",
        "@rmm",
        "@rules_cuda//cuda:runtime",
    ],
)
