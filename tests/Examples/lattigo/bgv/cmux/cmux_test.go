package main

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := cmux__configure()

	// Vector of plaintext values
	arg0 := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	arg1 := []int64{2, 3, 4, 5, 6, 7, 8, 9}
	cond := []bool{true, false, true, true, false, true, false, true}
	expected := []int64{1, 3, 3, 4, 6, 6, 8, 8}

	for i := 0; i < len(arg0); i++ {
		condEncrypted := cmux__encrypt__arg2(evaluator, params, ecd, enc, cond[i])

		resultCt := cmux(evaluator, params, ecd, arg0[i], arg1[i], condEncrypted)

		result := cmux__decrypt__result0(evaluator, params, ecd, dec, resultCt)

		if result != expected[i] {
			t.Errorf("Decryption error %d != %d", result, expected[i])
		}
	}
}
