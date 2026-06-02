# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# HEIR-local copy of @llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD.
#
# WHY THIS DIFFERS FROM UPSTREAM
# ------------------------------
# LLVM's llvm/lib/Support/Compression.cpp does `#include <zlib.h>` (angled). On
# a typical Linux host this is silently satisfied by /usr/include/zlib.h. Under
# HEIR's fully hermetic clang-22 toolchain there are no system include dirs, so
# the header must come from the @llvm_zlib repo, and the angled include fails
# with:
#   'zlib.h' file not found with <angled> include; use "quotes" instead
#
# Upstream relies on `strip_include_prefix = "."` on the `zlib` library. With
# the current Bazel/rules_cc that is a no-op: it produces neither a -I nor a
# _virtual_includes dir, so zlib.h lands only on the -iquote path (hence the
# "use quotes instead" diagnostic). Setting the `includes = ["."]` attribute
# instead emits a -isystem on the zlib library itself, BUT the consumer
# (llvm:Support) reaches zlib through @llvm-project//third-party:zlib, a
# `cc_library_wrapper` that re-exports via cc_common.merge_cc_infos. That merge
# propagates the transitive `includes` field (the -I from strip_include_prefix /
# virtual includes) but NOT `system_includes` (the -isystem from the `includes`
# attribute) -- verified empirically. So a -isystem on the zlib library never
# reaches Support, while zstd works only because its BUILD uses a non-trivial
# strip_include_prefix = "lib" that yields a propagating _virtual_includes -I.
#
# FIX: mirror zstd. We copy the public headers (zlib.h, zconf.h) into an
# `include/` subdir and expose them from a `zlib_headers` cc_library with
# `strip_include_prefix = "include"`. That generates a
# _virtual_includes/zlib_headers dir whose -I propagates through the wrapper to
# Support, resolving the angled `#include <zlib.h>`.
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for zlib)
    licenses = ["notice"],
)

copy_file(
    # The input template is identical to the CMake output.
    name = "zconf_gen",
    src = "zconf.h.in",
    out = "zconf.h",
    allow_symlink = True,
)

# Relocate the public headers into an `include/` subdirectory. This gives the
# zlib_headers library below a non-trivial strip_include_prefix ("include"),
# which is what makes the headers reach consumers. See the file-level comment
# for the full rationale (system_includes from the `includes` attribute are
# dropped by the //third-party:zlib cc_library_wrapper's merge_cc_infos; only
# the transitive `includes` field produced by strip_include_prefix survives,
# exactly as for the zstd BUILD which strips "lib").
copy_file(
    name = "zlib_h_relocate",
    src = "zlib.h",
    out = "include/zlib.h",
    allow_symlink = True,
)

copy_file(
    name = "zconf_h_relocate",
    src = ":zconf_gen",
    out = "include/zconf.h",
    allow_symlink = True,
)

# Public headers, exposed via a _virtual_includes/zlib_headers dir so that
# `#include <zlib.h>` (angled) resolves under the hermetic toolchain.
cc_library(
    name = "zlib_headers",
    hdrs = select({
        "@llvm-project//third-party:llvm_zlib_enabled": [
            "include/zconf.h",
            "include/zlib.h",
        ],
        "//conditions:default": [],
    }),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "zlib",
    srcs = select({
        "@llvm-project//third-party:llvm_zlib_enabled": [
            "adler32.c",
            "adler32_p.h",
            "chunkset.c",
            "chunkset_tpl.h",
            "compare258.c",
            "compress.c",
            "crc32.c",
            "crc32_comb.c",
            "crc32_comb_tbl.h",
            "crc32_p.h",
            "crc32_tbl.h",
            "deflate.c",
            "deflate.h",
            "deflate_fast.c",
            "deflate_medium.c",
            "deflate_p.h",
            "deflate_quick.c",
            "deflate_slow.c",
            "fallback_builtins.h",
            "functable.c",
            "functable.h",
            "infback.c",
            "inffast.c",
            "inffast.h",
            "inffixed_tbl.h",
            "inflate.c",
            "inflate.h",
            "inflate_p.h",
            "inftrees.c",
            "inftrees.h",
            "insert_string.c",
            "insert_string_tpl.h",
            "match_tpl.h",
            "trees.c",
            "trees.h",
            "trees_emit.h",
            "trees_tbl.h",
            "uncompr.c",
            "zbuild.h",
            "zendian.h",
            "zutil.c",
            "zutil.h",
            "zutil_p.h",
        ],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@llvm-project//third-party:llvm_zlib_enabled": [
            "zlib.h",
            ":zconf_gen",
        ],
        "//conditions:default": [],
    }),
    copts = [
        "-std=c11",
        "-DZLIB_COMPAT",
        "-DWITH_GZFILEOP",
        "-DWITH_OPTIM",
        "-DWITH_NEW_STRATEGIES",
        # For local builds you might want to add "-DWITH_NATIVE_INSTRUCTIONS"
        # here to improve performance. Native instructions aren't enabled in
        # the default config for reproducibility.
    ],
    defines = select({
        "@llvm-project//third-party:llvm_zlib_enabled": [
            "LLVM_ENABLE_ZLIB=1",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    # Re-export the public headers via their _virtual_includes/ path. This is
    # the transitive `includes` field (emitted as -I), which -- unlike the
    # `includes` attribute's system_includes -- survives the //third-party:zlib
    # cc_library_wrapper and thus reaches llvm:Support. See the file-level
    # comment above for the full rationale.
    deps = select({
        "@llvm-project//third-party:llvm_zlib_enabled": [":zlib_headers"],
        "//conditions:default": [],
    }),
)

alias(
    name = "zlib-ng",
    actual = ":zlib",
)
