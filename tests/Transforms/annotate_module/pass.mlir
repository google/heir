// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" %s | FileCheck %s --check-prefix=CHECK-CHEDDAR

// CHECK: module attributes {backend.openfhe, scheme.ckks}
// CHECK-CHEDDAR: module attributes {backend.cheddar, scheme.ckks}
module {

}
