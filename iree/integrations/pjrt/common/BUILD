# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.bzl", "iree_pjrt_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tensor_utils",
    srcs = [
        "tensor_utils.cc",
    ],
    hdrs = [
        "tensor_utils.h",
    ],
)

cc_test(
    name = "tensor_utils_test",
    srcs = [
        "tensor_utils_test.cc",
    ],
    linkstatic = True,
    deps = [
        ":tensor_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ]
)

iree_pjrt_cc_library(
    name = "impl",
    srcs = [
        "api_impl.cc",
        "platform.cc",
    ],
    hdrs = [
        "api_impl.h",
        "dylib_entry_point.cc.inc",
        "iree_helpers.h",
        "platform.h",
    ],
    deps = [
        ":compiler",
        ":debugging",
        ":tensor_utils",
        "@iree_core//runtime/src/iree/base",
        "@iree_core//runtime/src/iree/base:tracing",
        "@iree_core//runtime/src/iree/hal",
        "@iree_core//runtime/src/iree/modules/hal",
        "@iree_core//runtime/src/iree/vm",
        "@iree_core//runtime/src/iree/vm/bytecode:module",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

iree_pjrt_cc_library(
    name = "dylib_platform",
    srcs = [
        "dylib_platform.cc",
    ],
    hdrs = [
        "dylib_platform.h",
    ],
    deps = [
        ":debugging",
        ":impl",
        "@iree_core//compiler/bindings/c:headers",
        "@iree_core//compiler/bindings/c:loader",
        "@iree_core//runtime/src/iree/base/internal:file_io",
        "@iree_core//runtime/src/iree/base/internal:path",
    ],
)

iree_pjrt_cc_library(
    name = "compiler",
    srcs = [
        "compiler.cc",
    ],
    hdrs = [
        "compiler.h",
    ],
    deps = [
        ":debugging",
        "@iree_core//compiler/bindings/c:headers",
    ],
)

iree_pjrt_cc_library(
    name = "debugging",
    srcs = [
        "debugging.cc",
    ],
    hdrs = [
        "debugging.h",
    ],
)
