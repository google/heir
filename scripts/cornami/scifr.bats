#! /bin/env bats
@test "CFAIR Estimator (2)" {
  pushd ../../
  for name in $PWD/tests/Backend/cornami/and_gate.mlir $PWD/tests/Backend/cornami/lut_canonicalize.mlir
  do
    run  bazelisk run //tools:heir-opt -- --canonicalize --cggi-tigris-estimator $name
    [ "$status" -eq 0 ]
  done
  popd
}

@test "CFAIR OPT (1)" {
  pushd ../../
  for name in $PWD/tests/Backend/cornami/and_gate.mlir
  do
     BN=$(basename $name)
     echo "--- Running test $name ---"
     run bazelisk run //tools:heir-opt -- --canonicalize --cggi-tigris-estimator --cggi-to-scifrbool $name > /tmp/foo-$BN
     [ "$status" -eq 0 ]
     run bazelisk run //tools:heir-translate -- --emit-scifrbool  < `pwd`/tests/Backend/cornami/scifr_and.mlir
     [ "$status" -eq 0 ]
  done
  popd
}

@test "CKKS Estimator" {
  pushd ../../
  run bazelisk run //tools:heir-opt --  --ckks-tigris-estimator `pwd`/tests/Backend/cornami/ckksops.mlir
  [ "$status" -eq 0 ]
  popd
}

@test "Codegen for SCIFRBool AND gate" {
  pushd ../../
  run bazelisk run //tools:heir-opt --  --cggi-to-scifrbool `pwd`/tests/Backend/cornami/scifr_and.mlir
  [ "$status" -eq 0 ]
  popd
}
