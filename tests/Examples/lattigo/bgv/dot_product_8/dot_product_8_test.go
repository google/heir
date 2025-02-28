package dotproduct8

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := dot_product__configure()

	// Vector of plaintext values
	arg0 := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	arg1 := []int16{2, 3, 4, 5, 6, 7, 8, 9}

	expected := int16(240)

	ct0 := dot_product__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := dot_product__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := dot_product(evaluator, params, ecd, ct0, ct1)

	result := dot_product__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if result != expected {
		t.Errorf("Decryption error %d != %d", result, expected)
	}
}
