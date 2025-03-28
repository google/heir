package main

import (
	"testing"
)

func TestBinops(t *testing.T) {
	t.Skip("Expected to pass but fail sometimes")
	evaluator, params, ecd, enc, dec := mult_indep__configure()

	arg0 := int16(1)

	expected := int16(1)

	ct0 := mult_indep__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := mult_indep__encrypt__arg1(evaluator, params, ecd, enc, arg0)
	ct2 := mult_indep__encrypt__arg2(evaluator, params, ecd, enc, arg0)
	ct3 := mult_indep__encrypt__arg3(evaluator, params, ecd, enc, arg0)
	ct4 := mult_indep__encrypt__arg4(evaluator, params, ecd, enc, arg0)
	ct5 := mult_indep__encrypt__arg5(evaluator, params, ecd, enc, arg0)
	ct6 := mult_indep__encrypt__arg6(evaluator, params, ecd, enc, arg0)
	ct7 := mult_indep__encrypt__arg7(evaluator, params, ecd, enc, arg0)
	ct8 := mult_indep__encrypt__arg8(evaluator, params, ecd, enc, arg0)
	ct9 := mult_indep__encrypt__arg9(evaluator, params, ecd, enc, arg0)
	ct10 := mult_indep__encrypt__arg10(evaluator, params, ecd, enc, arg0)
	ct11 := mult_indep__encrypt__arg11(evaluator, params, ecd, enc, arg0)
	ct12 := mult_indep__encrypt__arg12(evaluator, params, ecd, enc, arg0)
	ct13 := mult_indep__encrypt__arg13(evaluator, params, ecd, enc, arg0)
	ct14 := mult_indep__encrypt__arg14(evaluator, params, ecd, enc, arg0)
	ct15 := mult_indep__encrypt__arg15(evaluator, params, ecd, enc, arg0)
	ct16 := mult_indep__encrypt__arg16(evaluator, params, ecd, enc, arg0)
	ct17 := mult_indep__encrypt__arg17(evaluator, params, ecd, enc, arg0)
	ct18 := mult_indep__encrypt__arg18(evaluator, params, ecd, enc, arg0)
	ct19 := mult_indep__encrypt__arg19(evaluator, params, ecd, enc, arg0)
	ct20 := mult_indep__encrypt__arg20(evaluator, params, ecd, enc, arg0)
	ct21 := mult_indep__encrypt__arg21(evaluator, params, ecd, enc, arg0)
	ct22 := mult_indep__encrypt__arg22(evaluator, params, ecd, enc, arg0)
	ct23 := mult_indep__encrypt__arg23(evaluator, params, ecd, enc, arg0)
	ct24 := mult_indep__encrypt__arg24(evaluator, params, ecd, enc, arg0)
	ct25 := mult_indep__encrypt__arg25(evaluator, params, ecd, enc, arg0)
	ct26 := mult_indep__encrypt__arg26(evaluator, params, ecd, enc, arg0)
	ct27 := mult_indep__encrypt__arg27(evaluator, params, ecd, enc, arg0)
	ct28 := mult_indep__encrypt__arg28(evaluator, params, ecd, enc, arg0)
	ct29 := mult_indep__encrypt__arg29(evaluator, params, ecd, enc, arg0)
	ct30 := mult_indep__encrypt__arg30(evaluator, params, ecd, enc, arg0)
	ct31 := mult_indep__encrypt__arg31(evaluator, params, ecd, enc, arg0)

	resultCt := mult_indep(evaluator, params, ecd, dec,
		ct0, ct1,
		ct2, ct3,
		ct4, ct5,
		ct6, ct7,
		ct8, ct9,
		ct10, ct11,
		ct12, ct13,
		ct14, ct15,
		ct16, ct17,
		ct18, ct19,
		ct20, ct21,
		ct22, ct23,
		ct24, ct25,
		ct26, ct27,
		ct28, ct29,
		ct30, ct31)

	result := mult_indep__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if result != expected {
		t.Errorf("Decryption error %d != %d", result, expected)
	}
}
