package leakyrelu

import (
	"math"
	"testing"
)

func TestLeakyrelu(t *testing.T) {
	evaluator, params, ecd, enc, dec := Leakyrelu__configure()

	const N = 64
	arg0 := make([]float32, N)
	expected := make([]float32, N)
	for i := 0; i < N; i++ {
		// Inputs in [-1, 1], the default approximation domain of max and min.
		x := math.Sin(float64(i) * 0.7)
		arg0[i] = float32(x)
		expected[i] = float32(math.Max(0, x) + 0.01*math.Min(0, x))
	}

	ct0 := Leakyrelu__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Leakyrelu(evaluator, params, ecd, ct0)
	result := Leakyrelu__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// max and min are approximated by a low-degree polynomial, whose error is
	// largest (about 0.04) near 0.
	errorThreshold := float64(0.1)
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
