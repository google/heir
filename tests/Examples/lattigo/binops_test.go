package binops

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := add__configure()

	// Vector of plaintext values
	// 0, 1, 2, 3
	arg0 := []int16{0, 1, 2, 3}
	// 1, 2, 3, 4
	arg1 := []int16{1, 2, 3, 4}

	expected := []int16{6, 15, 28, 1}

	ct0 := add__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := add__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := add(evaluator, params, ecd, ct0, ct1)

	result := add__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	for i := range 4 {
		if result[i] != expected[i] {
			t.Errorf("Decryption error at index %d: %d != %d", i, result[i], expected[i])
		}
	}
}
