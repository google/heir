package linalgdot

import (
	"math"
	"testing"
)

func TestLinalgDot(t *testing.T) {
	evaluator, params, ecd, enc, dec := dot_product__configure()

	// Vector of plaintext values
	arg0 := make([]float32, 1024)
	arg1 := make([]float32, 1024)
	for i := 0; i < 1024; i++ {
		arg0[i] = float32(i) / 1024.0
		arg1[i] = float32(i) / 1024.0
	}

	// Compute expected result
	expected := float32(0.0)
	for i := 0; i < 1024; i++ {
		expected += arg0[i] * arg1[i]
	}

	ct0 := dot_product__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := dot_product__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := dot_product(evaluator, params, ecd, ct0, ct1)

	result := dot_product__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.001)
	if math.Abs(float64(result[0]-expected)) > errorThreshold {
		t.Errorf("Decryption error %.4f != %.4f", result[0], expected)
	}
}
