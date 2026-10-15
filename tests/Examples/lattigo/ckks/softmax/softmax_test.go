package softmax

import (
	"math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	evaluator, params, ecd, enc, dec := Softmax__configure()

	// Input in [-1.0, 1.0]
	arg0 := []float32{-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0}

	// Compute expected exact softmax
	sumExp := float64(0.0)
	for _, val := range arg0 {
		sumExp += math.Exp(float64(val))
	}
	expected := make([]float32, len(arg0))
	for i, val := range arg0 {
		expected[i] = float32(math.Exp(float64(val)) / sumExp)
	}

	ct0 := Softmax__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	resultCt := Softmax(evaluator, params, ecd, ct0)

	result := Softmax__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// CGF-softmax is an approximation, so we use a larger error threshold.
	errorThreshold := float64(0.08) // 8%
	for i := 0; i < len(arg0); i++ {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > errorThreshold {
			t.Errorf("Index %d: Decryption error %.4f != %.4f (diff %.4f)", i, result[i], expected[i], diff)
		} else {
			t.Logf("Index %d: result %.4f, expected %.4f (diff %.4f)", i, result[i], expected[i], diff)
		}
	}
}
