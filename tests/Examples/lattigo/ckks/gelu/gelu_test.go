package gelu

import (
	"math"
	"testing"
)

func TestGelu(t *testing.T) {
	evaluator, params, ecd, enc, dec := Gelu__configure()

	const N = 64
	arg0 := make([]float32, N)
	expected := make([]float32, N)
	for i := 0; i < N; i++ {
		// Inputs in [-1.4, 1.4], so that x / sqrt(2) lies in the default
		// approximation domain [-1, 1] of erf.
		x := 1.4 * math.Sin(float64(i)*0.9)
		arg0[i] = float32(x)
		expected[i] = float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2)))
	}

	ct0 := Gelu__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Gelu(evaluator, params, ecd, ct0)
	result := Gelu__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	maxError := float64(0)
	for i := range expected {
		err := math.Abs(float64(result[i] - expected[i]))
		maxError = math.Max(maxError, err)
		if err > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
	t.Logf("Max error: %.6f", maxError)
}
