// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" %s | FileCheck %s

// CHECK: module attributes {backend.cheddar, scheme.ckks}
module {

}
