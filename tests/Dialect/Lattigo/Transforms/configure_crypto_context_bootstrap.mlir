// RUN: heir-opt --lattigo-configure-crypto-context %s | FileCheck %s

!bootstrapping_evaluator = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 17, Q = [1106058412451299513, 1056763241666817029, 957769724367225479, 919081519653443687, 1030837924888066153, 1084354410096143723, 1135846243351935917, 1087115004561311021, 997960547764032911, 892538949448853293, 1002528331340998513, 1100798419621231379, 981696679688787961, 1061922508412786269], P = [1152921504606846976], logDefaultScale = 60>, scheme.ckks} {
  func.func @bootstrap(%bootstrapping_evaluator: !bootstrapping_evaluator, %evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct) -> !ct {
    %ct_0 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %ct : (!bootstrapping_evaluator, !ct) -> !ct
    return %ct_0 : !ct
  }
}

// CHECK-DAG: ![[btEvalType:.*]] = !lattigo.ckks.bootstrapping_evaluator
// CHECK-DAG: ![[evalType:.*]] = !lattigo.ckks.evaluator

// CHECK: @bootstrap
// CHECK: @bootstrap__configure
// CHECK-SAME: -> (![[btEvalType]], ![[evalType]],
// CHECK: %[[param:.*]] = lattigo.ckks.new_parameters_from_literal
// CHECK: %[[btParams:.*]] = lattigo.ckks.new_bootstrapping_parameters_from_literal %[[param]] {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 14>}
// CHECK: %[[btEvalKeys:.*]] = lattigo.ckks.gen_evaluation_keys_bootstrapping %[[btParams]]
// CHECK: %[[btEval:.*]] = lattigo.ckks.new_bootstrapping_evaluator %[[btParams]], %[[btEvalKeys]]
