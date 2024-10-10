// RUN: heir-translate --import-autohog %S/toposort.json | FileCheck %s

// CHECK-LABEL: func.func private @toposort(
// CHECK: %[[v1:.*]] = cggi.or
// CHECK: %[[v2:.*]] = cggi.and
