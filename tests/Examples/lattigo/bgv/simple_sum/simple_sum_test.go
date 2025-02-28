package simplesum

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := simple_sum__configure()

	// Vector of plaintext values
	arg0 := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
		23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

	expected := int16(16 * 33)

	ct0 := simple_sum__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	resultCt := simple_sum(evaluator, params, ecd, ct0)

	result := simple_sum__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if result != expected {
		t.Errorf("Decryption error %d != %d", result, expected)
	}
}
