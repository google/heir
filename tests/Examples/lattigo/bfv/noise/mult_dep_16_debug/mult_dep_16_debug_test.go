package main

import (
	"testing"
)

func TestBinops(t *testing.T) {
	t.Skip("Expected to fail")
	evaluator, params, ecd, enc, dec := mult_dep__configure()

	arg0 := int16(1)

	expected := int16(1)

	ct0 := mult_dep__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	resultCt := mult_dep(evaluator, params, ecd, dec, ct0)

	result := mult_dep__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if result != expected {
		t.Errorf("Decryption error %d != %d", result, expected)
	}
}
