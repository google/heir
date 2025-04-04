package main

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := cross_level__configure()

	// Vector of plaintext values
	arg0 := []int16{1, 0, 1, 0}
	arg1 := []int16{0, 1, 0, 1}

	expected := []int16{1, 26, 1, 26}

	ct0 := cross_level__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := cross_level__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := cross_level(evaluator, params, ecd, dec, ct0, ct1)

	result := cross_level__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("Decryption error %d != %d", result[i], expected[i])
		}
	}
}
