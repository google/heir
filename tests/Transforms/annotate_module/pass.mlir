// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" %s | FileCheck %s

// CHECK: module attributes {backend.openfhe, scheme.ckks}
module {

}
