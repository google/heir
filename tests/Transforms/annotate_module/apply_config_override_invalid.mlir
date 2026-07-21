// RUN: not heir-opt --annotate-module="backend=openfhe" --apply-config-override="config=invalid_key=10" %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-KEY
// RUN: not heir-opt --annotate-module="backend=openfhe" --apply-config-override="config=bootstrapLevelsConsumed=abc" %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-VALUE

// CHECK-INVALID-KEY: Failed to parse value for key invalid_key: 10
// CHECK-INVALID-VALUE: Failed to parse value for key bootstrapLevelsConsumed: abc
module {

}
