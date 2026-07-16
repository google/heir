package padmatmul

import (
	"math"
	"testing"
)

func TestPadMatmul(t *testing.T) {
	btpEv, evaluator, params, ecd, enc, dec := Pad_matmul__configure()

	arg0 := make([]float32, 5*5)
	for i := 0; i < 25; i++ {
		arg0[i] = 1.0
	}
	arg1 := make([]float32, 5*5)
	for i := 0; i < 25; i++ {
		arg1[i] = 1.0
	}

	expected := make([]float32, 5*7)
	for i := 0; i < 5; i++ {
		for j := 0; j < 7; j++ {
			if j >= 2 {
				expected[i*7+j] = 4.0
			} else {
				expected[i*7+j] = 0.0
			}
		}
	}

	ct0 := Pad_matmul__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := Pad_matmul__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := Pad_matmul(btpEv, evaluator, params, ecd, ct0, ct1)

	result := Pad_matmul__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.001)
	for i := 0; i < 5*7; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
