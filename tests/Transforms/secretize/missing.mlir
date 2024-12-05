// RUN: heir-opt -secretize=function=foo -verify-diagnostics %s

// expected-error@+1 {{could not find function "foo" to secretize}}
module {
    func.func @comb(%a: i1, %b: i1) -> () {
        %0 = comb.truth_table %a, %b -> 6 : ui4
        return
    }
}
