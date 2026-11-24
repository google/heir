package prelu

import (
	"math"
	"testing"
)

func TestPrelu(t *testing.T) {
	evaluator, params, ecd, enc, dec := Prelu__configure()

	const N = 64
	arg0 := make([]float32, N)
	expected := make([]float32, N)
	for i := 0; i < N; i++ {
		// Inputs in [-1, 1], the default approximation domain of ReLU.
		x := math.Sin(float64(i) * 0.7)
		arg0[i] = float32(x)
		if x > 0 {
			expected[i] = float32(x)
		} else {
			expected[i] = float32(0.25 * x)
		}
	}

	ct0 := Prelu__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Prelu(evaluator, params, ecd, ct0)
	result := Prelu__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// PReLU is approximated through ReLU, whose low-degree polynomial has its
	// largest error (about 0.04) near 0, scaled here by 1 - 0.25.
	errorThreshold := float64(0.1)
	maxError := float64(0)
	for i := range expected {
		err := math.Abs(float64(result[i] - expected[i]))
		maxError = math.Max(maxError, err)
		if err > errorThreshold {
			t.Errorf("Decryption error at index %d (x=%.3f): %.4f != %.4f", i, arg0[i], result[i], expected[i])
		}
	}
	t.Logf("Max error: %.6f", maxError)
}
