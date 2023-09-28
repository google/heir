// RUN: heir-opt %s -verify-diagnostics

module {
    func.func @comb(%a: i1, %b: i1) -> () {
        %0 = comb.truth_table %a, %b -> [true, false, true, false]
        return
    }
}
