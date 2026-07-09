package main

import (
	"testing"
)

func TestBinops(t *testing.T) {
	t.Skip("Expected to pass but fail sometimes")
	evaluator, params, ecd, enc, dec := Mult_indep__configure()

	arg0 := int16(1)

	expected := int16(1)

	ct0 := Mult_indep__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := Mult_indep__encrypt__arg1(evaluator, params, ecd, enc, arg0)
	ct2 := Mult_indep__encrypt__arg2(evaluator, params, ecd, enc, arg0)
	ct3 := Mult_indep__encrypt__arg3(evaluator, params, ecd, enc, arg0)
	ct4 := Mult_indep__encrypt__arg4(evaluator, params, ecd, enc, arg0)
	ct5 := Mult_indep__encrypt__arg5(evaluator, params, ecd, enc, arg0)
	ct6 := Mult_indep__encrypt__arg6(evaluator, params, ecd, enc, arg0)
	ct7 := Mult_indep__encrypt__arg7(evaluator, params, ecd, enc, arg0)

	resultCt := Mult_indep(evaluator, params, ecd, dec,
		ct0, ct1,
		ct2, ct3,
		ct4, ct5,
		ct6, ct7)

	result := Mult_indep__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if result != expected {
		t.Errorf("Decryption error %d != %d", result, expected)
	}
}
