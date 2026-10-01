// RUN: heir-opt --annotate-module="backend=openfhe" --apply-config-override="config=bootstrapLevelsConsumed=10" %s | FileCheck %s

// CHECK: module attributes {backend.config_override = {bootstrapLevelsConsumed = 10 : i64}, backend.openfhe}
module {

}
