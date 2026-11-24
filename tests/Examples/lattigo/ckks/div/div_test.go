package div

import (
	"math"
	"testing"
)

func TestDiv(t *testing.T) {
	evaluator, params, ecd, enc, dec := Div__configure()

	const N = 64
	arg0 := make([]float32, N)
	arg1 := make([]float32, N)
	expected := make([]float32, N)
	for i := 0; i < N; i++ {
		arg0[i] = float32(math.Sin(float64(i) * 0.7))
		// Divisors in [0.5, 1.9], inside the default approximation domain
		// [0.1, 2.0] of the reciprocal.
		arg1[i] = float32(1.2 + 0.7*math.Cos(float64(i)*1.1))
		expected[i] = arg0[i] / arg1[i]
	}

	ct0 := Div__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := Div__encrypt__arg1(evaluator, params, ecd, enc, arg1)
	resultCt := Div(evaluator, params, ecd, ct0, ct1)
	result := Div__decrypt__result0(evaluator, params, ecd, dec, resultCt)

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
