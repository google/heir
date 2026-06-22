// RUN: heir-opt --preprocessing-to-openfhe --verify-diagnostics %s

// expected-warning@+1 {{split-preprocessing was run, but preprocessing-to-openfhe determined there are no plaintexts to preprocess.}}
module {
  func.func @no_preprocessing(%arg0: !preprocessing.storage<i32>) -> !preprocessing.storage<i32> {
    %storage = preprocessing.empty : !preprocessing.storage<i32>
    return %storage : !preprocessing.storage<i32>
  }
}
